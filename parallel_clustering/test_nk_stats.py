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
  times = [0] * 9
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
  return times

def ps(lst):
  for pair in lst:
    sys.stdout.write(str(pair) + ", ")
    sys.stdout.flush()
  sys.stdout.write("\n")
  sys.stdout.flush()

def outputs(modularities, precisions, recalls, num_comms, min_comms, max_comms, avg_comms):
  ps(modularities)
  ps(precisions)
  ps(recalls)
  ps(num_comms)
  ps(min_comms)
  ps(max_comms)
  ps(avg_comms)

# cluster time, total time, modularity, precision, recall, # comm,
# min comm, max comm, avg comm
def main():
  files = ["amazon.edges","dblp.edges","lj.edges"] #,"orkut.edges"
  pre = ["amazon", "dblp", "lj"] #, "orkut"
  res_float_lj = [0.02 * ((1 + 1.0 / 5.0) ** x) for x in range(0, 90)]
  res_lj = [str(x) for x in res_float_lj]
  res_float = [0.02 * ((1 + 1.0 / 5.0) ** x) for x in range(0, 101)]
  res = [str(x) for x in res_float]
  modularities = []
  precisions = []
  recalls = []
  num_comms = []
  min_comms = []
  max_comms = []
  avg_comms = []
  for idx, filename in enumerate(files):
    sys.stdout.write(filename + "\n")
    sys.stdout.flush()
    use_res = res_lj if idx == 2 else res
    for r in use_res:
      ss = ("bazel run //clustering:cluster-stats_main -- --"
      "input_graph=/home/jeshi/snap/" + filename + " --input_clusters="
      "/home/jeshi/out/"+ pre[idx] +"_" + r + "_nk --clusterer_config='correlation_clusterer_config"
      " {resolution: " + r + "}' --input_communities"
      "='/home/jeshi/snap/com-" + pre[idx] + ".top5000.cmty.txt'")
      out = shellGetOutput(ss)
      pair = format2(out)
      modularities.append(pair[2])
      precisions.append(pair[3])
      recalls.append(pair[4])
      num_comms.append(pair[5])
      min_comms.append(pair[6])
      max_comms.append(pair[7])
      avg_comms.append(pair[8])
    outputs(modularities, precisions, recalls, num_comms, min_comms, max_comms, avg_comms)
    modularities[:] = []
    precisions[:] = []
    recalls[:] = []
    num_comms[:] = []
    min_comms[:] = []
    max_comms[:] = []
    avg_comms[:] = []
    sys.stdout.write("\n\n")
    sys.stdout.flush()

if __name__ == "__main__":
  main()
