import os
import sys
import signal
import time
import subprocess

def signal_handler(signal,frame):
  print("bye\n")
  sys.exit(0)
signal.signal(signal.SIGINT,signal_handler)

def shellGetOutput(str) :
  process = subprocess.Popen(str,shell=True,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  output, err = process.communicate()
  
  if (len(err) > 0):
    print(str+"\n"+output+err)
  return output

def format2(out):
  times = 0
  for line in out.splitlines():
    line = line.strip()
    split = [x.strip() for x in line.split(':')]
    if split[0].startswith("Time"):
      times += float(split[1])
  return times

def ps(lst):
  for pair in lst:
    sys.stdout.write(str(pair) + ", ")
    sys.stdout.flush()
  sys.stdout.write("\n")
  sys.stdout.flush()

def outputs(count_times):
  ps(count_times)

# cluster time, total time, modularity, precision, recall, # comm,
# min comm, max comm, avg comm
def main():
  files = ["amazon.edges","dblp.edges","lj.edges","orkut.edges"]
  pre = ["amazon", "dblp", "lj", "orkut"]
  res_float = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  res = [str(x) for x in res_float]
  count_times = []
  for idx, filename in enumerate(files):
    sys.stdout.write(filename + "\n")
    sys.stdout.flush()
    for r in res:
      ss = ("python3.7 nk.py /home/jeshi/snap/" + filename + " /home/"
      "jeshi/out_nk/" + pre[idx] + "_" + r + "_nk " + r)
      out = shellGetOutput(ss)
      pair = format2(out)
      count_times.append(pair)
    outputs(count_times)
    count_times[:] = []
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    del count_times[:]

if __name__ == "__main__":
  main()
