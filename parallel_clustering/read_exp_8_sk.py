import numpy as np
import os
import requests
import tempfile
import time
import math

import sklearn

def community_to_cluster_id(filename):
  arr = np.zeros([1] , dtype=int)
  idx = 0
  with open(filename) as fp:
    for line in fp:
      keys = [x.strip() for x in line.split('\t')]
      for x in keys:
        y = int(x)
        if y + 1 > arr.size:
          np.resize(arr, (y + 1))
        arr[y] = idx
      idx += 1
  return arr


def main():
  programs = ["ParallelCorrelationClusterer","CorrelationClusterer"]
  programs_pres = ["pc","c"]
  files = ["cancer_wh","digits_wh","iris_wh", "letter_wh","olivetti_wh","wine_wh"]
  pres = ["cancer","digits","iris","letter","olivetti","wine"]
  pf = "letter"
  asy = "true"
  ref = "true"
  mp = "nbhr"
  all_iter = ["false", "true"]
  resolutions = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  nw = "60"
  read_dir = "/home/jeshi/snap/"
  community = read_dir + "com-" + pres[file_idx] + ".top5000.cmty.txt"
  write_dir = "/home/jeshi/clustering_out_exp8_weighted/"
  cluster_true = community_to_cluster_id(community)
  length = len(resolutions)
  nmis = [0]*length
  aris = [0]*length
  accs = [0]*length
  # read community
  # then read out_filename
  # compute acc, etc for each r
  for prog_idx, prog in enumerate(programs):
    for ai in all_iter:
      if (ai == "true") and (prog_idx == 0):
        continue
      print("Prog: " + prog + ", AI: " + ai)
      for r_idx, r in enumerate(resolutions):
        out_filename = write_dir + programs_pres[prog_idx] + "_" + pf + "_" + ai + "_" + str(r) + "_" + asy + "_" + ref + "_" + mp+"_" + str(nw) + ".cluster"
        cluster_here = community_to_cluster_id(out_filename)
        nmis[r_idx] = sklearn.metrics.normalized_mutual_info_score(cluster_true, cluster_here)
        aris[r_idx] = sklearn.metrics.adjusted_rand_score(cluster_true, cluster_here)
        accs[r_idx] = sklearn.metrics.accuracy_score(cluster_true, cluster_here)
      for r_idx, x in enumerate(resolutions):
        print(str(x) + "\t"),
        print(str(nmis[r_idx]) + "\t"),
        print(str(aris[r_idx]) + "\t"),
        print(str(accs[r_idx]))


if __name__ == "__main__":
  main()
