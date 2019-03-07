"""""An interface to work with the database

Attributes:
    arxivid_docid (dict): dict with arxivids and the id of the document in the db. Is used to find papers by their arxivid
    db (TinyDB): The TinyDB database with all papers
    DB_PATH (str): File to store the db in
    serialization (DateTimeSerializer): a custom serializer to support datetime objects
"""
import numpy as np

from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware

from mydate import DateTimeSerializer
import sys

##############################INITIALIZE THE DATABASE##############################
DB_PATH = 'data/db.json'
readPath=True
if readPath:
  DB_PATH = open("PATHS.txt").readline().strip()
sys.stderr.write("Warning! DB_PATH is: %s\n"%(DB_PATH))

#Tinydb database with a custom serializer to support datetime objects.


serialization = SerializationMiddleware()
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
db = TinyDB(DB_PATH, storage=serialization)


def get_arxivid_docid_dict():
    """Iterates over the whole db and crates a dict with the arxivids of a paper and
    the document id in the dataset. 
    
    Returns:
        dict: dict with arxivids and the id of the document in the db
    """
    arxivid_docid = {}
    for paper in db:
        arxivid_docid[paper['arxivid']] =  paper.doc_id
    return arxivid_docid

arxivid_docid = get_arxivid_docid_dict()






##############################FUNCTIONS TO WORK WITH THE DATABASE##############################
def add_paper(paper):
    """Adds a single paper to the database.
    If a paper with the same arxiv id is already in the db, the paper will be updated and not inserted.
    
    Args:
        paper (dict): a dict containing the keys arxivid,created,citations, authors, title, abstract 
    """
    if paper['arxivid'] not in arxivid_docid:
        did = db.insert(paper)
        arxivid_docid[paper['arxivid']] = did
    else:
        update_paper(paper['arxivid'],paper)

def update_paper(arxivid,fields):
    """Updates a paper in the db by its arxivid
    
    Args:
        arxivid (str): the id the a paper has on arxiv
        fields (dict): a dict containing a fields of the paper to update
    """
    db.update(fields, doc_ids=[arxivid_docid[arxivid]])

def remove_paper(arxivid):
    """
        remove one paper from the db
    Args:
        arxivid (str): the id the a paper has on arxiv
    """
    db.remove(doc_ids=[arxivid_docid[arxivid]])
def get_paper(arxivid):
    """Returns a paper in the db
    
    Args:
        arxivid (str): the id the apaper has on arxiv
    
    Returns:
        dict: The paper as a dict
    """
    return db.get(doc_id = arxivid_docid[arxivid])

def find_paper():
    """TODO:implement 
    """
    pass

def get_papers_after(startdate):
    """Gets all papers created after a date
    
    Args:
        startdate (datetime.datetime): startdate
    
    Returns:
        list: all papers creted after and on startdate
    """
    query = Query()
    return db.search(query.created >= startdate)

def get_papers_in_timewindow(startdate, enddate):
    """Gets all papers created between two dates
    
    Args:
        startdate (datetime.datetime): startdate
        enddate (datetime.datetime): enddate
    
    Returns:
        list: all papers creted after and on startdate and before and on enddate
    """

    query = Query()
    return db.search(query.created.test(lambda d: startdate <= d <= enddate))

def get_all_papers_iterator():
    """Returns an iterator object for all papers in the dataset
    
    Returns:
        iterator: Iterator over all papers in the db
    """
    return iter(db)

def get_arxivid_embedding(embedding ='infersent', textpart ='abstract'):
    """returns a dict with arxiv ids and their infersent embedding. 
    Only includes papers if their embeddings had been calculated for that textpart
    
    Args:
        embedding (str): infersent or unisent
        textpart (str, optional): abstract or title
    
    Returns:
        dict: dict with arxiv ids and their infersent embedding
    """
    arxivid_embedding = {}
    with open('data/%s_%s.csv'%(embedding,textpart)) as f:
        for line in f:
            l = line.strip().split()
            id = l[0]
            embedding = np.array([float(v) for v in l[1:]])
            arxivid_embedding[id] = embedding
    return arxivid_embedding

