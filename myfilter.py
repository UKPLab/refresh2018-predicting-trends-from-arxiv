import sys

# SAMPLE USAGE:
# python3 myfilter.py 4 2017 12 < top1000_csCL.dat

header=True
i=0
minVal=int(sys.argv[1])
yr=int(sys.argv[2])
mnth=int(sys.argv[3])

for line in sys.stdin:
  if i==0 and header:
    i+=1
  else:
    line = line.strip()
    x = line.split("\t")
    citations,created = x[2],x[4]
    _,c = citations.split(":")
    c = int(c.strip())
    #print(c,created)
    created = created.split(":")
    #print(created)
    date = created[1].strip().split()[0]
    year,month,day = [int(y) for y in date.split("-")]
    if year>yr: continue
    if year==yr and month>mnth: continue
    if c<minVal: continue
    print(line)
