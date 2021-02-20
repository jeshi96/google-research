import os
import sys
import signal
import time
import subprocess
from math import sqrt

def main():
  pres = "amazon" #["amazon","dblp","lj","orkut","friendster"]
  async_sync = ["true", "false"]
  refines = ["true", "false"]
  resolutions = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.55, 0.85, 1, 1.25, 1.5, 1.75, 2]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  nw = num_workers[0]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_new/"
  length = len(resolutions)
  avg_num_clusters = [0]*length
  avg_obj = [0]*length
  std_dev_obj = [0]*length
  avg_disagreement_obj = [0]*length
  avg_time  = [0]*length
  std_dev_time = [0]*length
  avg_num_inner = [0]*length
  avg_num_outer = [0]*length
  num_rounds = 4
  for asy in async_sync:
    for ref in refines:
      print("Graph: " + pres + ", Async: " + asy + ", Refine: " + ref)
      for r_idx, r in enumerate(resolutions):
        read_filename = write_dir + pres + "_" + str(r) + "_" + asy + "_" + ref + "_" + str(nw) + ".out"
        with open(read_filename, "r") as read_file:
          num_clusters = [0]*num_rounds
          objs = [0]*num_rounds
          disagreement_objs = [0]*num_rounds
          times = [0] * num_rounds 
          num_inners = [0] * num_rounds
          num_outers = [0] * num_rounds
          idx = num_rounds - 1
          for line in read_file:
            line = line.strip()
            split = [x.strip() for x in line.split(':')]
            if split[0].startswith("Read Time"):
              idx += 1
              idx = idx % num_rounds
            elif split[0].startswith("Num inner"):
              num_inners[idx] += float(split[1])
            elif split[0].startswith("Num outer"):
              num_outers[idx] += float(split[1])
            elif split[0].startswith("# PBBS time"):
              times[idx] = float(split[3])
            elif split[0].startswith("Objective"):
              objs[idx] = float(split[1])
            elif split[0].startswith("Disagreement Objective"):
              disagreement_objs[idx] = float(split[1])
            elif split[0].startswith("Number of Clusters"):
              num_clusters[idx] = float(split[1])
            avg_num_clusters[r_idx] = sum(num_clusters) / num_rounds
            avg_obj[r_idx] = sum(objs) / num_rounds
            avg_disagreement_obj[r_idx] = sum(disagreement_objs) / num_rounds
            avg_time[r_idx] = sum(times) / num_rounds
            avg_num_inner[r_idx] = sum(num_inners) / num_rounds
            avg_num_outer[r_idx] = sum(num_outers) / num_rounds
            std_dev_obj[r_idx] = sqrt(sum([(x - avg_obj[r_idx]) ** 2 for x in objs]) / num_rounds)
            std_dev_time[r_idx] = sqrt(sum([(x - avg_time[r_idx]) ** 2 for x in times]) / num_rounds)
      # now we must output
      for r_idx, r in enumerate(resolutions):
        print(str(r), end = "\t")
        print(avg_time[r_idx], end = "\t")
        print(std_dev_time[r_idx], end = "\t")
        print(avg_obj[r_idx], end = "\t")
        print(std_dev_obj[r_idx], end = "\t")
        print(avg_disagreement_obj[r_idx], end = "\t")
        print(avg_num_clusters[r_idx], end = "\t")
        print(avg_num_inner[r_idx], end = "\t")
        print(avg_num_outer[r_idx], end = "\n")

if __name__ == "__main__":
  main()