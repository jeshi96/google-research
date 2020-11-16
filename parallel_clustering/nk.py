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
      print(file=fp)
  # input snap communities
  # input_comm = sys.argv[2]
  # comms = []
  # with open(input_comm) as fp:
  #   lines = fp.readlines()
  #   for line in lines:
  #     comms.append([int(x) for x in line.split('\t')])
  # # for each comm in plmCommunities, find largest intersect with comms
  # precision_tot = 0.0
  # recall_tot = 0.0
  # for comm in comms:
  #   max_size = 0.0
  #   plm_size = 0.0
  #   comm_size = len(comm)
  #   for i in plmCommunities.getSubsetIds():
  #     plm = plmCommunities.getMembers(i)
  #     inter = intersection(plm, comm)
  #     if (len(inter) > max_size):
  #       max_size = len(inter)
  #       plm_size = len(plm)
  #   precision_tot += max_size / plm_size
  #   if comm_size == 0:
  #     assert max_size == 0
  #   if comm_size > 0:
  #     recall_tot += max_size / comm_size
  # precision_tot /= len(comms)
  # recall_tot /= len(comms)
  # print("Precision: " + str(precision_tot))
  # print("Recall: " + str(recall_tot))

if __name__ == "__main__":
  main()