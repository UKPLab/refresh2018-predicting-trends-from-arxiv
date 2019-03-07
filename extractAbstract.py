# python3 extractAbstract.py data/dbs/12312018/db_cs* < topPapers/top1000.cs_CL_12302018 |less


import sys
import json

def retrieveAbstracts(h,dta):
  for x in dta:
    arxivid = dta[x]["arxivid"]
    abstract = dta[x]["abstract"]
    h[arxivid] = abstract
  return h 

h={}

for file in sys.argv[1:]:
  f=open(file)
  data = json.load(f)
  dta = data["_default"]
  h = retrieveAbstracts(h,dta)

header=True
collected=0

for i,line in enumerate(sys.stdin):
  line = line.strip()
  if i==0 and header==True: continue
  x = line.split("\t")
  arxivid = x[0]
  title = x[1]
  cits = x[2]
  zscore = x[3]
  cits = int(cits.split(":")[-1])
  abstract = h[arxivid].replace("\n"," ")
  if cits>=4:
    print("\t".join([str(arxivid),title,str(cits),str(zscore),abstract,"None","None"]))
    collected+=1
  if collected>=100: break 
