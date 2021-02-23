import os
import sys
import signal
import time
import subprocess
from math import sqrt

def read_and_print(read_filename, num_rounds,avg_obj, avg_time, std_dev_obj, std_dev_time, r_idx):
  with open(read_filename, "r") as read_file:
    objs = [0]*num_rounds
    times = [0] * num_rounds
    idx = num_rounds - 1
    for line in read_file:
      line = line.strip()
      split = [x.strip() for x in line.split(':')]
      if split[0].startswith("Read Time"):
        idx += 1
        idx = idx % num_rounds
      elif split[0].startswith("# PBBS time"):
        times[idx] += float(split[3])
      elif split[0].startswith("Objective"):
        objs[idx] = float(split[1])
      avg_obj[r_idx] = sum(objs) / num_rounds
      avg_time[r_idx] = sum(times) / num_rounds
      std_dev_obj[r_idx] = sqrt(sum([(x - avg_obj[r_idx]) ** 2 for x in objs]) / num_rounds)
      std_dev_time[r_idx] = sqrt(sum([(x - avg_time[r_idx]) ** 2 for x in times]) / num_rounds)

def main():
  programs = ["ParallelCorrelationClusterer","ParallelModularityClusterer"]
  programs_pres = ["pc","pm"]
  files = ["amazon_h","orkut_h"]#"dblp_h", "lj_h","orkut_h","friendster_h"]
  pres = ["amazon","orkut"]#"dblp","lj","orkut","friendster"]
  async_sync = ["true", "false"]
  refines = ["false"]
  moves = ["ALL_MOVE"]
  moves_pres = ["all"]
  resolutions = [0.01, 0.85]#[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out/"
  length = len(files)
  avg_num_clusters = [0]*length
  avg_obj = [0]*length
  std_dev_obj = [0]*length
  avg_time  = [0]*length
  std_dev_time = [0]*length
  num_rounds = 4
  for prog_idx, prog in enumerate(programs):
    for r in resolutions:
      for ref_idx, ref in enumerate(refines):
        for asy_idx, asy in enumerate(async_sync):
          for move_idx, move in enumerate(moves):
            for nw in num_workers:
              print("Prog: " + prog + ", Resolution: " + str(r) + ", Refine: " + str(ref) + ", Async: " + str(asy) + "Move: " + str(move))
              for file_idx, filename in enumerate(files):
                read_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                read_and_print(read_filename, num_rounds, avg_obj, avg_time, std_dev_obj, std_dev_time, file_idx)
              # now we must output
              for r_idx, r in enumerate(files):
                print(str(pres[r_idx]), end = "\t")
                print(avg_time[r_idx], end = "\t")
                print(std_dev_time[r_idx], end = "\t")
                print(avg_obj[r_idx], end = "\t")
                print(std_dev_obj[r_idx], end = "\t")


if __name__ == "__main__":
  main()