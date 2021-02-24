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
  files = ["cancer_h","digits_h","iris_h", "letter_h","olivetti_h","wine_h"]
  pres = ["cancer","digits","iris","letter","olivetti","wine"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  all_iter = ["false", "true"]
  resolutions = [x * (1.0/100.0) for x in range(1, 100)]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp8_weighted_pr/"
  length = len(resolutions)
  avg_num_clusters = [0]*length
  avg_obj = [0]*length
  avg_time  = [0]*length
  precision = [0]*length
  recall = [0]*length
  num_rounds = 1
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for ai in all_iter:
        if (ai == "true") and (prog_idx == 0):
          continue
        print("Prog: " + prog + ", File: " + pres[file_idx] + ", All iter: " + ai)
        for ref_idx, ref in enumerate(refines):
          for asy_idx, asy in enumerate(async_sync):
            for move_idx, move in enumerate(moves):
              for nw in num_workers:
                for r_idx, r in enumerate(resolutions):
                  read_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                  read_and_print(read_filename, num_rounds, avg_obj, avg_time, r_idx, precision, recall)
                # now we must output
                for r_idx, x in enumerate(resolutions):
                  print(str(x) + "\t"),
                  print(str(avg_time[r_idx]) + "\t"),
                  print(str(avg_obj[r_idx]) + "\t"),
                  print(str(precision[r_idx]) + "\t"),
                  print(str(recall[r_idx]))


if __name__ == "__main__":
  main()