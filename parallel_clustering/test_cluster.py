import os
import sys
import signal
import time
import subprocess

def signal_handler(signal,frame):
  print "bye\n"
  sys.exit(0)
signal.signal(signal.SIGINT,signal_handler)

def shellGetOutput(str) :
  process = subprocess.Popen(str,shell=True,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  output, err = process.communicate()
  
  
  #if (len(err) > 0):
  #    raise NameError(str+"\n"+output+err)
  return output

def format2(out):
  times = [0] * 12
  for line in out.splitlines():
    line = line.strip()
    split = [x.strip() for x in line.split(':')]
    if split[0].startswith("Modularity"):
      times[2] += float(split[1])
    elif split[0].startswith("Cluster Time"):
      times[1] += float(split[1])
    elif split[0].startswith("Avg precision"):
      times[3] += float(split[1])
    elif split[0].startswith("Avg recall"):
      times[4] += float(split[1])
    elif split[0].startswith("Num comm"):
      times[5] += float(split[1])
    elif split[0].startswith("Min size"):
      times[6] += float(split[1])
    elif split[0].startswith("Max size"):
      times[7] += float(split[1])
    elif split[0].startswith("Avg size"):
      times[8] += float(split[1])
    elif split[0].startswith("Read Time"):
      times[0] += float(split[1])
    elif split[0].startswith("Num in core"):
      times[9] += float(split[1])
    elif split[0].startswith("Num inner"):
      times[10] += float(split[1])
    elif split[0].startswith("Num outer"):
      times[11] += float(split[1])
  return times

def ps(lst):
  for pair in lst:
    sys.stdout.write(str(pair) + ", ")
    sys.stdout.flush()
  sys.stdout.write("\n")
  sys.stdout.flush()

def outputs(count_times, cluster_times, modularities, precisions, recalls, num_comms, min_comms, max_comms, avg_comms, num_in_core, num_inner, num_outer):
  ps(count_times)
  ps(cluster_times)
  ps(modularities)
  ps(precisions)
  ps(recalls)
  ps(num_comms)
  ps(min_comms)
  ps(max_comms)
  ps(avg_comms)
  ps(num_in_core)
  ps(num_inner)
  ps(num_outer)

# cluster time, total time, modularity, precision, recall, # comm,
# min comm, max comm, avg comm
def main():
  files = ["amazon.edges","dblp.edges"]#"lj.edges","orkut.edges"]
  pre = ["amazon", "dblp"]# "lj", "orkut"]
  res_float = [0.02 * ((1 + 1.0 / 5.0) ** x) for x in range(0, 101)]
  res = [str(x) for x in res_float]
  cluster_method = ["LOUVAIN"] #"DEFAULT_CLUSTER_MOVES", 
  kcores = [3, 5, 7]
  fix_kcores = ["0", "1"]
  count_times = []
  cluster_times = []
  modularities = []
  precisions = []
  recalls = []
  num_comms = []
  min_comms = []
  max_comms = []
  avg_comms = []
  num_in_core = []
  num_inner = []
  num_outer = []
  for idx, filename in enumerate(files):
    sys.stdout.write(filename + "\n")
    sys.stdout.flush()
    for c in cluster_method:
      sys.stdout.write(c + "\n")
      sys.stdout.flush()
      for k in kcores:
        for f in fix_kcores:
          sys.stdout.write("Core: "+str(k) + "\n")
          sys.stdout.flush()
          sys.stdout.write("Fix: "+f+"\n")
          sys.stdout.flush()
          for r in res:       
            ss = ("bazel run //clustering:cluster-in-memory_main -- --"
            "input_graph=/home/jeshi/snap/" + filename + " --clusterer_name="
            "ParallelModularityClusterer --clusterer_config='correlation_clusterer_config"
            " {resolution: " + r + ", subclustering_method: NONE, "
            "clustering_moves_method: " + c + ", preclustering_method: KCORE, "
            "kcore_config {kcore_cutoff: "+str(k)+", fix_core_clusters: "+f+"}}' --input_communities"
            "='/home/jeshi/snap/com-" + pre[idx] + ".top5000.cmty.txt'")
            out = shellGetOutput(ss)
            pair = format2(out)
            count_times.append(pair[0])
            cluster_times.append(pair[1])
            modularities.append(pair[2])
            precisions.append(pair[3])
            recalls.append(pair[4])
            num_comms.append(pair[5])
            min_comms.append(pair[6])
            max_comms.append(pair[7])
            avg_comms.append(pair[8])
            num_in_core.append(pair[9])
            num_inner.append(pair[10])
            num_outer.append(pair[11])
          outputs(count_times, cluster_times, modularities, precisions, recalls, num_comms, min_comms, max_comms, avg_comms, num_in_core, num_inner, num_outer)
          count_times[:] = []
          cluster_times[:] = []
          modularities[:] = []
          precisions[:] = []
          recalls[:] = []
          num_comms[:] = []
          min_comms[:] = []
          max_comms[:] = []
          avg_comms[:] = []
          num_in_core[:] = []
          num_inner[:] = []
          num_outer[:] = []
          sys.stdout.write("\n\n")
          sys.stdout.flush()

if __name__ == "__main__":
  main()
