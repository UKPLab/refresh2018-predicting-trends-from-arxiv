import json
import re
import pandas as pd
import urllib
import sys
from datetime import date
def semantic_scholar_api(id):
    """ Uses the sematic scholar api to get metadata and citation infrmation from semantic scholar
    
    Args:
        id (TYPE): arxiv id
    
    Returns:
        dict: metadata as a dict. None if paper not found
    """
    while True:
        try:
            query = "http://api.semanticscholar.org/v1/paper/arXiv:"+id
            print(query)
            with urllib.request.urlopen(query) as url:
                data = json.loads(url.read().decode())
                data['arxiv_id'] = id
                if 'error' in data:
                    return None
                break
        except urllib.error.URLError as e:
            print("paper not found")
            return None

    return data




