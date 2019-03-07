"""crawls papers starting from a certain date and update the citattion information starting from a certain date
Attributes:
    parser (argparse): parses command line arguments
"""
import datetime
import argparse

import numpy as np

from get_arxiv_data2 import harvest
from get_semanticscholar_data import semantic_scholar_api
import datahandler2 as datahandler
from mydate import valid_date

parser = argparse.ArgumentParser(
    description='Crawl papers from arxiv and metadata from semantic scholar or just update the metadata')
parser.add_argument('-np', "--newpapersstartdate", help="The Start Date to harvest new papers from - format YYYY-MM-DD",
                    type=valid_date, default=None)
parser.add_argument('-um', "--updatemetastartdate",
                    help="The Start Date update paperes metadata (citations, z-score)- format YYYY-MM-DD",
                    type=valid_date, default=None)
parser.add_argument('-cat', "--category", type=str, help="category on arxiv", default="cs.LG")


def main():
    """Usage: python3 update_dataset.py -h
    """
    args = parser.parse_args()
    if args.newpapersstartdate:
        harvest_new_papers(args.newpapersstartdate, category=args.category)
        filter_double_citations(args.newpapersstartdate)
    if args.updatemetastartdate:
        update_db_citations(args.updatemetastartdate)
        filter_double_citations(args.updatemetastartdate)

    filter_wrong_papers()


def harvest_new_papers(startdate, category="cs.LG"):
    """harvestes new papers from arxiv, gets their citation counts and adds them to the db

    Args:
        startdate (datetime.datetime): startdate
        category (str, optional): category on arxiv(default "cs.LG")
    """
    today = datetime.date.today()
    papers = harvest(arxiv="cs", startdate=startdate.strftime("%Y-%m-%d"), enddate=today.strftime("%Y-%m-%d"))
    for paper in papers:
        if category in paper['categories'] and paper['created'] >= startdate:
            data = semantic_scholar_api(paper['arxivid'])
            if data and (paper['created'].year <= data['year']):
                paper['authors'] = data['authors']
                paper['year'] = data['year']
                paper['citations'] = []
                for c in data['citations']:
                    print(c.keys(),c["paperId"],paper['arxivid'])
                    try:
                        c['year'] = int(c['year'])
                    except TypeError:
                        continue
                    if c['year'] >= paper['created'].year:
                        paper['citations'].append(c)
                    else:  # If the paper has citations prior to the created date it has been published already somewhere else and will not be safed
                        break
                else:
                    datahandler.add_paper(paper)

    for paper in datahandler.get_papers_after(startdate):
        if 'citations' in paper:
            paper['z-score'] = calculate_z_score(paper)
            datahandler.update_paper(paper['arxivid'], paper)


def update_db_citations(startdate):
    """Updates citation counts and z-score for each paper in the db created before and on startdate

    Args:
        startdate (datetime.datetime): startdate
    """
    papers = datahandler.get_papers_after(startdate)
    for paper in papers:
        data = semantic_scholar_api(paper['arxivid'])
        if data:
            paper['citations'] = []
            for c in data['citations']:
                try:
                    c['year'] = int(c['year'])
                except TypeError:
                    continue
                if c['year'] >= paper['created'].year:
                    paper['citations'].append(c)
            datahandler.update_paper(paper['arxivid'], paper)

    for paper in papers:
        if 'citations' in paper:
            paper['z-score'] = calculate_z_score(paper)
            datahandler.update_paper(paper['arxivid'], paper)


def calculate_z_score(paper, timewindow_days=10):
    """calculates the z-score for a paper using papers gatherd from the db created in the same timewindow around (+-timewindow_days) 
    the creation of the paper

    z-score defined by newman:
    "we take the count of citations received by a paper, subtract the
    mean for papers published around the same time, and
    divide by the standard deviation"
    Args:
        paper (dict): paper in form of a dict containing the keys arxivid,created,citations, authors
        timewindow_days (int, optional): timewindow of days to use as a context to calculate the z score(default 10)
    
    Returns:
        float: z-score of the paper
    """
    startdate = paper['created'] - datetime.timedelta(days=timewindow_days)
    enddate = paper['created'] + datetime.timedelta(days=timewindow_days)
    papers_in_timewindow = datahandler.get_papers_in_timewindow(startdate, enddate)

    ncitations = len(paper['citations'])
    citations = [len(p['citations']) for p in papers_in_timewindow]

    std = np.std(citations)
    mean = np.mean(citations)
    z = 0
    try:
        z = (ncitations - mean) / std
    except ZeroDivisionError:
        z = (ncitations - mean)
    if z != z:
        z = 0
    return z


def filter_double_citations(startdate=None):
    """ removes citations from a paper if they have the same title as another citation for the same papers

    Args:
        startdate (datetime.datetime): startdate. None to filter all papers
    """
    if startdate:
        papers = datahandler.get_papers_after(startdate)
    else:
        papers = datahandler.get_all_papers_iterator()

    n, p = 0, 0
    for paper in papers:
        titles = set()
        filtered_citations = []
        for citation in paper['citations']:
            if not citation['title'] in titles:
                titles.add(citation['title'])
                filtered_citations.append(citation)
            else:
                n += 1

        if len(filtered_citations) != len(paper['citations']):
            paper['citations'] = filtered_citations
            p += 1
            datahandler.update_paper(paper['arxivid'], paper)
    print(p, n)


def filter_wrong_papers():
    """
        removes papers from the database
        if there is an older paper with the same title and authors
        or citations with a year previous to the publication year

    """
    title_arxivid = {}
    npaper = 0
    papers_to_remove = set()
    for paper in datahandler.get_all_papers_iterator():
        npaper += 1
        if not paper['title'].lower() in title_arxivid:
            title_arxivid[paper['title'].lower()] = paper['arxivid']
        else:
            duplicate, toremove = is_duplicate(title_arxivid[paper['title'].lower()], paper['arxivid'])
            if duplicate:
                papers_to_remove.add(toremove['arxivid'])
                # print(toremove['title'],toremove['created'] )
                # datahandler.remove_paper(toremove['arxivid'])

        for citation in paper['citations']:
            if citation['year'] < paper['created'].year:
                papers_to_remove.add(paper['arxivid'])
                # print("year", paper['title'], paper['created'], citation['title'], citation_date, len(paper['citations']))
                # if 'z-score' in paper:
                # print(paper['z-score'])

            """
            #should remove papers with citation previous to publication date.
            #Does not work, becaus it is unclear weather the date of the citation or the paper is wrong
            elif citation['title'].lower() in title_arxivid:
                found += 1
                citation_date = datahandler.get_paper(title_arxivid[citation['title'].lower()])['created']
                if citation_date < paper['created']:
                    print(paper['title'], paper['created'],citation['title'],citation_date, len(paper['citations']))
                    filter = True
                    if 'z-score' in paper:
                        print(paper['z-score'])

            else:
                notfound += 1
            """
    for arxivid in papers_to_remove:
        datahandler.remove_paper(arxivid)


def is_duplicate(arxivid1, arxivid2):
    """ checks weather two papers have the same author and title

    Args:
        arxivid1(str): arxivid of the first paper
        arxivid2(str): arxivid of the second paper

    Returns:
        either True and the newer paper if they are duplicates
        or False,None if they are no duplicates


    """
    p1 = datahandler.get_paper(arxivid1)
    p2 = datahandler.get_paper(arxivid2)
    if p1['title'].lower() == p2['title'].lower():
        p1names = set([a['name'] for a in p1['authors']])
        p2names = set([a['name'] for a in p2['authors']])
        if p1names == p2names:
            if p1['created'] < p2['created']:
                return True, p2
            else:
                return True, p1

    return False, None


if __name__ == "__main__":
    main()
