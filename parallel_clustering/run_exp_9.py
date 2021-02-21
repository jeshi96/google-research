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

def run_9():
  programs = ["ParallelCorrelationClusterer", "ParallelModularityClusterer", "CorrelationClusterer", "ModularityClusterer"]
  programs_pres = ["pc","pm","c","m"]
  files = ["hyperlink2012_sym.bytepd","hyperlink2014_sym.bytepd"]#,"dblp_h", "lj_h","orkut_h","friendster_h"]
  pres = ["h12","h14"]#,"dblp","lj","orkut","friendster"]
  async_sync = ["true"]
  refines = ["true"]
  moves = ["NBHR_MOVE"]
  moves_pres = ["nbhr"]
  resolutions = [0.01, 0.85]#[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  num_workers = [160]
  read_dir = "/home/jeshi/snap/"
  write_dir = "/home/jeshi/clustering_out_exp9/"
  for prog_idx, prog in enumerate(programs):
    for file_idx, filename in enumerate(files):
      for r in resolutions:
        for ref_idx, ref in enumerate(refines):
          for asy_idx, asy in enumerate(async_sync):
            for move_idx, move in enumerate(moves):
              for nw in num_workers:
                if True: #for i in range(4):
                  out_filename = write_dir + programs_pres[prog_idx] + "_" + pres[file_idx] + "_" + str(r) + "_" + asy + "_" + ref + "_" + moves_pres[move_idx]+"_" + str(nw) + ".out"
                  ss = ("NUM_THREADS="+str(nw)+" timeout 6h bazel run //clustering:cluster-in-memory_main -- --"
                  "input_graph=" + read_dir  + filename + " --clusterer_name=" + prog + " "
                  " --clusterer_config='correlation_clusterer_config"
                  " {resolution: " + str(r) + ", subclustering_method: NONE_SUBCLUSTERING, "
                  "clustering_moves_method: LOUVAIN , preclustering_method: NONE_PRECLUSTERING, "
                  "refine: "+ref+", async: "+asy+", move_method: "+move+"}'")
                  out = shellGetOutput(ss)
                  appendToFile(out, out_filename)
