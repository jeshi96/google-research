import networkit as nk
import sys
import os
import time

def intersection(lst1, lst2):
  return list(set(lst1) & set(lst2))

def main():
  filename = sys.argv[1]
  G = nk.readGraph(filename, nk.Format.EdgeListTabZero)
  x = float(sys.argv[3])
  start = time.time()
  plmCommunities = nk.community.detectCommunities(G, algo=nk.community.PLM(G, True, gamma=x))
  end = time.time()
  print("Time: " + str(end-start))
  output_fp = sys.argv[2]
  with open(output_fp, 'w') as fp:
    for i in plmCommunities.getSubsetIds():
      plm = plmCommunities.getMembers(i)
      for p in plm:
        print(str(p),end='\t',flush=True,file=fp)
      print(flush=True,file=fp)

if __name__ == "__main__":
  main()