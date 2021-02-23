import os
import sys
import signal
import time
import subprocess
from math import sqrt

def read_and_print(read_filename, num_rounds,avg_obj, avg_time, std_dev_obj, std_dev_time, r_idx, use_mod):
  if (use_mod):
    obj_str = "Modularity"
  else:
    obj_str = "Objective"
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
      elif split[0].startswith(obj_str):
        objs[idx] = float(split[1])
      avg_obj[r_idx] = sum(objs) / num_rounds
      avg_time[r_idx] = sum(times) / num_rounds
      std_dev_obj[r_idx] = sqrt(sum([(x - avg_obj[r_idx]) ** 2 for x in objs]) / num_rounds)
      std_dev_time[r_idx] = sqrt(sum([(x - avg_time[r_idx]) ** 2 for x in times]) / num_rounds)

def main():
  programs = ["ParallelCorrelationClusterer", "ParallelModularityClusterer"]
  programs_pres = ["pc","pm"]
  #files = [100, 1000, 1]#,"dblp_h", "lj_h","orkut_h","friendster_h"]
  #pres = ["friendster","twitter"]#,"dblp","lj","orkut","friendster"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  resolutions = [0.01, 0.85]#[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  num_workers = [60]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp7_new/"
  length = len(num_workers)
  avg_num_clusters = [0]*length
  avg_obj = [0]*length
  std_dev_obj = [0]*length
  avg_time  = [0]*length
  std_dev_time = [0]*length
  num_rounds = 4
  for prog_idx, prog in enumerate(programs):
    for ref_idx, ref in enumerate(refines):
      for asy_idx, asy in enumerate(async_sync):
        for move_idx, move in enumerate(moves):
          for r2_idx, r in enumerate(resolutions):
            for r_idx, nw in enumerate(num_workers):
            for y in range(2):
              print("Prog: " + prog + ", Res: " + r + ", Rmat: " + str(y + 1))
              for x in range(6):
                if y == 0:
                  xx = 10**(x + 3)
                else:
                  xx = 10**(x + 2)
                if y == 0:
                  filename = "rmat_" + str(xx)
                  pre_file = "rmat" + str(xx)
                else:
                  filename = "rmat2_" + str(xx)
                  pre_file = "rmat2" + str(xx)
                use_mod = (program_pres[prog_idx] == "pm")
                read_filename = write_dir + programs_pres[prog_idx] + "_" + pre_file + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                read_and_print(read_filename, num_rounds, avg_obj, avg_time, std_dev_obj, std_dev_time, r_idx, use_mod)
              # now we must output
              for r_idx in range(6):
                print(str(r_idx) + "\t"),
                print(str(avg_time[r_idx]) + "\t"),
                print(str(std_dev_time[r_idx]) + "\t"),
                print(str(avg_obj[r_idx]) + "\t"),
                print(str(std_dev_obj[r_idx]))


if __name__ == "__main__":
  main()