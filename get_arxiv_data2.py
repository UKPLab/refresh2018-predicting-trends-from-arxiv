
"""havest metadata from arxiv
source http://betatim.github.io/posts/analysing-the-arxiv/
Harvestes metadata from arxiv in order to find all papers in a category.
Attributes:
    ARXIV (str): link to the arxiv oai api
    OAI (str): link to oai
"""

import time
import urllib
import datetime
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET


import numpy as np
import sys
import re


OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"

def harvest(arxiv="cs", startdate = "2000-01-01", enddate = "2017-12-31"): #physics:hep-ex
    """
    Harvestes metadata for a specific category on arxiv
    
    Args:
        arxiv (str, optional): category on arxiv (cs, physics:hep-ex)
    
    Returns:
        pandas dataframe: a dataframe with metadata harvested from arxiv
    """

    papers = []
    base_url = "http://export.arxiv.org/oai2?verb=ListRecords&"
    url = (base_url +
           "from=%s&until=%s&"%(startdate,enddate) +
           "metadataPrefix=arXiv&set=%s"%arxiv)

    maxG=10
    i=0   
 
    while True:
        print( "fetching", url)
        try:
            response = urllib.request.urlopen(url)
            
        except urllib.error.HTTPError as e:
            if e.code == 503:
                to = int(e.hdrs.get("retry-after", 30))
                print("Got 503. Retrying after {0:d} seconds.".format(to))

                time.sleep(to)
                continue
                
            else:
                raise
            
        xml = response.read()

        root = ET.fromstring(xml)

        for record in root.find(OAI+'ListRecords').findall(OAI+"record"):
            arxiv_id = record.find(OAI+'header').find(OAI+'identifier')
            meta = record.find(OAI+'metadata')
            info = meta.find(ARXIV+"arXiv")
            created = info.find(ARXIV+"created").text
            created = datetime.datetime.strptime(created, "%Y-%m-%d")
            categories = info.find(ARXIV+"categories").text
            #print(ET.tostring(info))
            authors = []
            for author in info.find(ARXIV+"authors").findall(ARXIV+"author"):
                a= {}

                a['keyname'] = author.find(ARXIV+"keyname").text
                try:
                    a['forenames'] = author.find(ARXIV+'forenames').text
                except AttributeError as e:
                    a['forenames'] = ''
                authors.append(a)
            # if there is more than one DOI use the first one
            # often the second one (if it exists at all) refers
            # to an eratum or similar
            doi = info.find(ARXIV+"doi")
            if doi is not None:
                doi = doi.text.split()[0]
            arxivid = info.find(ARXIV+"id").text
            arxivid = re.sub('/','',arxivid)
            contents = {'title': info.find(ARXIV+"title").text,
                        'arxivid': arxivid,
                        'abstract': info.find(ARXIV+"abstract").text.strip(),
                        'created': created,
                        'categories': categories.split(),
                        'doi': doi,
                        'authors' : authors
                        }

            papers.append(contents)

        # The list of articles returned by the API comes in chunks of
        # 1000 articles. The presence of a resumptionToken tells us that
        # there is more to be fetched.
        token = root.find(OAI+'ListRecords').find(OAI+"resumptionToken")
        if token is None or token.text is None:
            break

        else:
            url = base_url + "resumptionToken=%s"%(token.text)
        if i>maxG:
          break
        i+=1
            
    return papers



    
