import datahandler
import datetime,sys
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, KMeans
from mydate import valid_date

import argparse

parser = argparse.ArgumentParser("prints the top k papers by z-score starting from startdate")
parser.add_argument("startdate", help="format YYYY-MM-DD", type=valid_date)
parser.add_argument("k", type=int)
parser.add_argument("-min", "--min_citations", help="papers with less than min_citations are filterd out", type=int,
                    default=0)
parser.add_argument("-pred", "--top_predictions", help="print papers with top predictions instead of z-scores",
                    default=False, action='store_true')


def main():
    args = parser.parse_args()
    print(args)
    if args.top_predictions:
        top_prediction(args.startdate, k=args.k, print_output=True, textpart="abstract")
    else:
        top_zscore(args.startdate, k=args.k, mincitations=args.min_citations, print_output=True)


def top_zscore(startdate, k=10, print_output=False, mincitations=4):
    """return the top k papers by z-score ignoring papers published before Startdate or less than mincitations
    
    Args:
        startdate (datetime.datetime): startdate
        k (int, optional): how many papers are returned
        print_output (bool, optional): If true output is print in a formatted way
        mincitations (int, optional): Ignore papers with less than mincitations
    
    Returns:
        list:list of tupels with arxiv id and z-scores 
    """
    arxivid_z = {}
    for paper in datahandler.get_papers_after(startdate):
        if 'z-score' in paper and len(paper['citations']) >= mincitations:
            arxivid_z[paper['arxivid']] = paper['z-score']

    top = sorted(arxivid_z.items(), key=(lambda x: x[1]), reverse=True)
    if print_output:
        for arxivid, z in top[:k]:
            paper = datahandler.get_paper(arxivid)
            #print(paper.keys(),paper["created"],paper["year"])
            created = paper["created"]
            sys.stdout.write("%s\t%s\tnum_citations:%i\tzscore: %f\t" % (
            paper['arxivid'], paper['title'].replace('\n', ' '), len(paper['citations']), z))
            print("created:",created)
            # print("\n")

    return top[:k]


def top_prediction(startdate, k=10, print_output=False, textpart="abstract"):
    """return the top k papers by prediction ignoring papers published before Startdate or less than mincitations

    Args:
        startdate (datetime.datetime): startdate
        k (int, optional): how many papers are returned
        print_output (bool, optional): If true output is print in a formatted way

    Returns:
        list:list of tupels with arxiv id and predictions
    """
    field = "prediction_%s" % textpart
    arxivid_pred = {}
    for paper in datahandler.get_papers_after(startdate):
        if field in paper:
            arxivid_pred[paper['arxivid']] = paper[field]

    top = sorted(arxivid_pred.items(), key=(lambda x: x[1]), reverse=True)
    if print_output:
        for arxivid, z in top[:k]:
            paper = datahandler.get_paper(arxivid)
            print("%s\t%s\tnum_citations:%i\tprediction: %f" % (
                paper['arxivid'], paper['title'].replace('\n', ' '), len(paper['citations']), z))
            # print("\n")

    return top[:k]


def top_similarity(arxivid, k=10, print_output=False):
    """Returns the most similar papers by cosine similarity on abstract embeddigs.
    
    Args:
        arxivid (str): arxivid 
        k (int, optional): how many similar papers
        print_output (bool, optional): If true output is printed in a formatted way
    
    Returns:
        list: list of tupels with arxiv id and cosine similarity
    """
    arxivid_embedding = datahandler.get_arxivid_embedding(textpart='abstract')

    if not arxivid in arxivid_embedding:
        print('no embedding for %s' % arxivid)
        return False

    emb = arxivid_embedding[arxivid]
    arxivid_similarity = {}
    for id, embedding in arxivid_embedding.items():
        arxivid_similarity[id] = cosine_similarity(np.array([emb, embedding])).tolist()[0][1]
    top = sorted(arxivid_similarity.items(), key=(lambda x: x[1]), reverse=True)

    if print_output:
        print("PAPERS SIMILAR TO:\n%s\n" % datahandler.get_paper(arxivid)['title'])
        for arxivid, sim in top[:k]:
            paper = datahandler.get_paper(arxivid)
            print("%s\t%s\tcosinesimilarity: %f\n" % (paper['arxivid'], paper['title'].replace('\n', ' '), sim))
            print("\n")

    return top[:k]


def cluster_papers(arxivids=None, n_clusters=20, print_output=True):
    """Cluster papers by their abstract infersent embeddings
    
    Args:
        arxivids (list, optional): list of arxivids to cluster or if None all papers with abstract infersent embeddings are clusterd
        n_clusters (int, optional): number of clusters
        print_output (bool, optional):  If true output is printed in a formatted way
    
    Returns:
        dict: dict with arxivids as keys and their clusters as int as value
    """
    arxivid_embedding = datahandler.get_arxivid_embedding(textpart='abstract')
    arxivid_embedding = {k: v for k, v in arxivid_embedding.items() if (not arxivids or (k in arxivids))}

    i = 0
    index_arxivid = {}
    x = []
    for id, emb in arxivid_embedding.items():
        x.append(emb)
        index_arxivid[i] = id
        i += 1
    x = np.array(x)

    # clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='complete')
    clustering = KMeans(n_clusters=n_clusters)
    y = clustering.fit_predict(x).tolist()

    arxivid_cluster = {}
    for i, label in enumerate(y):
        arxivid_cluster[index_arxivid[i]] = label

    if print_output:
        cluster_papers = defaultdict(list)
        for id, label in arxivid_cluster.items():
            cluster_papers[label].append(datahandler.get_paper(id))

        allzscores = []
        for cluster, papers in cluster_papers.items():
            z_scores = []
            print('CLUSTER:%i\n' % cluster)
            for paper in papers:
                z = paper['z-score']
                z_scores.append(z)
                print("%s\t%s\tnum_citations:%i\tz-score: %f" % (
                paper['arxivid'], paper['title'].replace('\n', ' '), len(paper['citations']), z))
            print("MEAN Z-SCORE:%f\n\n" % np.mean(z_scores))
            allzscores += z_scores
        print("MEAN Z-SCORE OVER ALL SAMPLES:%f\n\n" % np.mean(allzscores))

    return arxivid_cluster


if __name__ == "__main__":
    main()


# get the top 100 papers by z-score and cluster them
# top_z = top_zscore(startdate = datetime.datetime(2017,5,1), k =100,print_output = True)
# top_p = top_prediction(startdate = datetime.datetime(2017,6,1), k =100,print_output = True)
# arxivid_cluster = cluster_papers([x[0] for x in top], n_clusters = 20,print_output = True)
# intersection = set(dict(top_z).keys()).intersection(set(dict(top_p).keys()))
# print(intersection)
# print(len(intersection))
