import os
import sys
import signal
import time
import subprocess
from math import sqrt

def read_and_print(read_filename, num_rounds,avg_obj, avg_time, r_idx, precision, recall):
  with open(read_filename, "r") as read_file:
    for line in read_file:
      line = line.strip()
      split = [x.strip() for x in line.split(':')]
      if split[0].startswith("# PBBS time"):
        avg_time[r_idx] += float(split[3])
      elif split[0].startswith("Objective"):
        avg_obj[r_idx] = float(split[1])
      elif split[0].startswith("Avg precision"):
        precision[r_idx] = float(split[1])
      elif split[0].startswith("Avg recall"):
        recall[r_idx] = float(split[1])

def main():
  programs = ["ParallelCorrelationClusterer","CorrelationClusterer"]
  programs_pres = ["pc","c"]
  files = ["amazon_h","orkut_h"]#"dblp_h", "lj_h","orkut_h","friendster_h"]
  pres = ["amazon","orkut"]#"dblp","lj","orkut","friendster"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  resolutions = [x * (1.0/100.0) for x in range(1, 100)]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp2/"
  length = len(resolutions)
  avg_num_clusters = [0]*length
  avg_obj = [0]*length
  avg_time  = [0]*length
  precision = [0]*length
  recall = [0]*length
  num_rounds = 1
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      print("Prog: " + prog + ", File: " + pres[file_idx])
      for ref_idx, ref in enumerate(refines):
        for asy_idx, asy in enumerate(async_sync):
          for move_idx, move in enumerate(moves):
            for nw in num_workers:
              for r_idx, r in enumerate(resolutions):
                read_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                read_and_print(read_filename, num_rounds, avg_obj, avg_time, r_idx, precision, recall)
              # now we must output
              for r_idx, x in enumerate(resolutions):
                print(str(x), end = "\t")
                print(avg_time[r_idx], end = "\t")
                print(avg_obj[r_idx], end = "\t")
                print(precision[r_idx], end = "\t")
                print(recall[r_idx], end = "\n")


if __name__ == "__main__":
  main()