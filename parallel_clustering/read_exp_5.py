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
      elif split[0].startswith("Modularity"):
        objs[idx] = float(split[1])
      avg_obj[r_idx] = sum(objs) / num_rounds
      avg_time[r_idx] = sum(times) / num_rounds
      std_dev_obj[r_idx] = sqrt(sum([(x - avg_obj[r_idx]) ** 2 for x in objs]) / num_rounds)
      std_dev_time[r_idx] = sqrt(sum([(x - avg_time[r_idx]) ** 2 for x in times]) / num_rounds)

def main():
  programs = ["ModularityClusterer"]
  programs_pres = ["m"]
  files = ["amazon_h","dblp_h", "lj_h","orkut_h"]
  pres = ["amazon","dblp","lj","orkut"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  resolutions = ["1e-05", "0.0001", "0.001", "0.01", "0.1", "0.25", "0.5", "0.75", "0.8", "0.85", "0.9", "0.95", "0.99"]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  #read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp5_rerun/"
  length = len(resolutions)
  avg_num_clusters = [0]*length
  avg_obj = [0]*length
  std_dev_obj = [0]*length
  avg_time  = [0]*length
  std_dev_time = [0]*length
  num_rounds = 1
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for ref_idx, ref in enumerate(refines):
        for asy_idx, asy in enumerate(async_sync):
          for move_idx, move in enumerate(moves):
            for nw in num_workers:
              print("Prog: " + prog + ", File: " + str(pres[file_idx]))
              for r_idx, r in enumerate(resolutions):
                read_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                read_and_print(read_filename, num_rounds, avg_obj, avg_time, std_dev_obj, std_dev_time, r_idx)
              # now we must output
              for r_idx, x in enumerate(resolutions):
                print(str(x) + "\t"),
                print(str(avg_time[r_idx]) + "\t"),
                print(str(std_dev_time[r_idx]) + "\t"),
                print(str(avg_obj[r_idx]) + "\t"),
                print(str(std_dev_obj[r_idx]))


if __name__ == "__main__":
  main()
