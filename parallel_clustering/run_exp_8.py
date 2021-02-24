import os
import sys
import signal
import time
import subprocess

def signal_handler(signal,frame):
  print "bye\n"
  sys.exit(0)
signal.signal(signal.SIGINT,signal_handler)

def shellGetOutput(str1) :
  process = subprocess.Popen(str1,shell=True,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  output, err = process.communicate()
  
  if (len(err) > 0):
    print(str1+"\n"+output+err)
  return output

def appendToFile(out, filename):
  with open(filename, "a+") as out_file:
    out_file.writelines(out)

def run_8_corr_weighted_pr():
  programs = ["ParallelCorrelationClusterer","CorrelationClusterer"]
  programs_pres = ["pc","c"]
  files = ["cancer_wh","digits_wh","iris_wh", "letter_wh","olivetti_wh","wine_wh"]
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
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for r in resolutions:
        for ref_idx, ref in enumerate(refines):
          for asy_idx, asy in enumerate(async_sync):
            for move_idx, move in enumerate(moves):
              for nw in num_workers:
                for ai in all_iter:
                  if (ai == "true") and (prog_idx == 0):
                    continue
                  for i in range(1):
                    out_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                    ss = ("NUM_THREADS="+str(nw)+" timeout 6h bazel run //clustering:cluster-in-memory_main -- --"
                    "input_graph=" + read_dir  + filename + " --clusterer_name=" + prog + " "
                    " --float_weighted=true --clusterer_config='correlation_clusterer_config"
                    " {resolution: " + str(r) + ", subclustering_method: NONE_SUBCLUSTERING, "
                    "clustering_moves_method: LOUVAIN , preclustering_method: NONE_PRECLUSTERING, "
                    "all_iter: "+ai+", refine: "+ref+", async: "+asy+", move_method: "+move+"}' --input_communities"
                    "='" + read_dir + "com-" + pres[file_idx] + ".top5000.cmty.txt'")
                    out = shellGetOutput(ss)
                    appendToFile(out, out_filename)

def run_8_corr_weighted():
  programs = ["ParallelCorrelationClusterer","CorrelationClusterer"]
  programs_pres = ["pc","c"]
  files = ["cancer_wh","digits_wh","iris_wh", "letter_wh","olivetti_wh","wine_wh"]
  pres = ["cancer","digits","iris","letter","olivetti","wine"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  all_iter = ["false", "true"]
  resolutions = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp8_weighted/"
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for r in resolutions:
        for ref_idx, ref in enumerate(refines):
          for asy_idx, asy in enumerate(async_sync):
            for move_idx, move in enumerate(moves):
              for nw in num_workers:
                for ai in all_iter:
                  if (ai == "true") and (prog_idx == 0):
                    continue
                  for i in range(4):
                    out_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                    out_cluster_fn = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_"+str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".cluster"
                    ss = ("NUM_THREADS="+str(nw)+" timeout 6h bazel run //clustering:cluster-in-memory_main -- --"
                    "input_graph=" + read_dir  + filename + " --clusterer_name=" + prog + " "
                    " --float_weighted=true --output_clustering="+out_cluster_fn+" --clusterer_config='correlation_clusterer_config"
                    " {resolution: " + str(r) + ", subclustering_method: NONE_SUBCLUSTERING, "
                    "clustering_moves_method: LOUVAIN , preclustering_method: NONE_PRECLUSTERING, "
                    "all_iter: "+ai+", refine: "+ref+", async: "+asy+", move_method: "+move+"}'")
                    out = shellGetOutput(ss)
                    appendToFile(out, out_filename)

def run_8_corr():
  programs = ["ParallelCorrelationClusterer","CorrelationClusterer"]
  programs_pres = ["pc","c"]
  files = ["cancer_h","digits_h","iris_h", "letter_h","olivetti_h","wine_h"]
  pres = ["cancer","digits","iris","letter","olivetti","wine"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  all_iter = ["false", "true"]
  resolutions = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp8/"
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for r in resolutions:
        for ref_idx, ref in enumerate(refines):
          for asy_idx, asy in enumerate(async_sync):
            for move_idx, move in enumerate(moves):
              for nw in num_workers:
                for ai in all_iter:
                  if (ai == "true") and (prog_idx == 0):
                    continue
                  for i in range(4):
                    out_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                    out_cluster_fn = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_"+str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".cluster"
                    ss = ("NUM_THREADS="+str(nw)+" timeout 6h bazel run //clustering:cluster-in-memory_main -- --"
                    "input_graph=" + read_dir  + filename + " --clusterer_name=" + prog + " "
                    " --output_clustering="+out_cluster_fn+" --clusterer_config='correlation_clusterer_config"
                    " {resolution: " + str(r) + ", subclustering_method: NONE_SUBCLUSTERING, "
                    "clustering_moves_method: LOUVAIN , preclustering_method: NONE_PRECLUSTERING, "
                    "all_iter: "+ai+", refine: "+ref+", async: "+asy+", move_method: "+move+"}'")
                    out = shellGetOutput(ss)
                    appendToFile(out, out_filename)

def run_8_mod():
  programs = ["ParallelModularityClusterer","ModularityClusterer"]
  programs_pres = ["pm","m"]
  files = ["cancer_h","digits_h","iris_h", "letter_h","olivetti_h","wine_h"]
  pres = ["cancer","digits","iris","letter","olivetti","wine"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  all_iter = ["false", "true"]
  resolutions = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp8/"
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for r in resolutions:
        for ref_idx, ref in enumerate(refines):
          for asy_idx, asy in enumerate(async_sync):
            for move_idx, move in enumerate(moves):
              for nw in num_workers:
                for ai in all_iter:
                  if (ai == "true") and (prog_idx == 0):
                    continue
                  for i in range(4):
                    out_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                    out_cluster_fn = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_"+str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".cluster"
                    ss = ("NUM_THREADS="+str(nw)+" timeout 6h bazel run //clustering:cluster-in-memory_main -- --"
                    "input_graph=" + read_dir  + filename + " --clusterer_name=" + prog + " "
                    " --output_clustering="+out_cluster_fn+" --clusterer_config='correlation_clusterer_config"
                    " {resolution: " + str(r) + ", subclustering_method: NONE_SUBCLUSTERING, "
                    "clustering_moves_method: LOUVAIN , preclustering_method: NONE_PRECLUSTERING, "
                    "all_iter: "+ai+", refine: "+ref+", async: "+asy+", move_method: "+move+"}'")
                    out = shellGetOutput(ss)
                    appendToFile(out, out_filename)




def run_8_corr_pr():
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
  write_dir = "/home/jeshi/clustering_out_exp8_pr/"
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for r in resolutions:
        for ref_idx, ref in enumerate(refines):
          for asy_idx, asy in enumerate(async_sync):
            for move_idx, move in enumerate(moves):
              for nw in num_workers:
                for ai in all_iter:
                  if (ai == "true") and (prog_idx == 0):
                    continue
                  for i in range(1):
                    out_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                    ss = ("NUM_THREADS="+str(nw)+" timeout 6h bazel run //clustering:cluster-in-memory_main -- --"
                    "input_graph=" + read_dir  + filename + " --clusterer_name=" + prog + " "
                    " --clusterer_config='correlation_clusterer_config"
                    " {resolution: " + str(r) + ", subclustering_method: NONE_SUBCLUSTERING, "
                    "clustering_moves_method: LOUVAIN , preclustering_method: NONE_PRECLUSTERING, "
                    "all_iter: "+ai+", refine: "+ref+", async: "+asy+", move_method: "+move+"}' --input_communities"
                    "='" + read_dir + "com-" + pres[file_idx] + ".top5000.cmty.txt'"
                    out = shellGetOutput(ss)
                    appendToFile(out, out_filename)

def run_8_mod_pr():
  programs = ["ParallelModularityClusterer","ModularityClusterer"]
  programs_pres = ["pm","m"]
  files = ["cancer_h","digits_h","iris_h", "letter_h","olivetti_h","wine_h"]
  pres = ["cancer","digits","iris","letter","olivetti","wine"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  all_iter = ["false", "true"]
  resolutions = [0.02 * ((1 + 1.0 / 5.0) ** x) for x in range(0, 101)]
  num_workers = [60]#[1, 2, 4, 8, 16, 30, 60]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp8_pr/"
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for r in resolutions:
        for ref_idx, ref in enumerate(refines):
          for asy_idx, asy in enumerate(async_sync):
            for move_idx, move in enumerate(moves):
              for nw in num_workers:
                for ai in all_iter:
                  if (ai == "true") and (prog_idx == 0):
                    continue
                  for i in range(1):
                    out_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + ai + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                    ss = ("NUM_THREADS="+str(nw)+" timeout 6h bazel run //clustering:cluster-in-memory_main -- --"
                    "input_graph=" + read_dir  + filename + " --clusterer_name=" + prog + " "
                    " --clusterer_config='correlation_clusterer_config"
                    " {resolution: " + str(r) + ", subclustering_method: NONE_SUBCLUSTERING, "
                    "clustering_moves_method: LOUVAIN , preclustering_method: NONE_PRECLUSTERING, "
                    "all_iter: "+ai+", refine: "+ref+", async: "+asy+", move_method: "+move+"}' --input_communities"
                    "='" + read_dir + "com-" + pres[file_idx] + ".top5000.cmty.txt'"
                    out = shellGetOutput(ss)
                    appendToFile(out, out_filename)


