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

def main():
  pre = ["amazon", "dblp", "lj", "orkut"]
  nn = [334863, 317080, 3997962, 3072441]
  tect_dir = "/home/jeshi/tectonic_graphs/"
  read_dir = "/home/jeshi/snap/"
  out_file = "/home/jeshi/tectonic_graphs/out"
  for idx, p in enumerate(pre):
    sys.stdout.write("\n" + p + "\n")
    sys.stdout.flush()
    tect_p = tect_dir + p
    s1 = ("python relabel-graph.py "+read_dir+"com-"+p+".ungraph."
    "txt "+read_dir+"com-"+p+".top5000.cmty.txt "+tect_p+".mace "+tect_p+".communities")
    s2 = ("/home/jeshi/mace/mace C -l 3 -u 3 "+tect_p+".mace "+tect_p+".triangles")
    s3 = ("python mace-to-list.py "+tect_p+".mace "+tect_p+".edges")
    s4 = ("./tree-clusters "+tect_p+".weighted "+str(nn[idx])+" > "+tect_p+".clusters")
    s5 = ("python2 grade-clusters.py "+tect_p+".communities "+tect_p+".clusters "+tect_p+".grading")
    start1 = time.time()
    out1 = shellGetOutput(s1)
    end1 = time.time()
    sys.stdout.write("Time 1: " + str(end1 - start1) + "\n")
    sys.stdout.flush()
    appendToFile(out1, out_file)
    start2 = time.time()
    out2 = shellGetOutput(s2)
    end2 = time.time()
    sys.stdout.write("Time 2: " + str(end2 - start2) + "\n")
    sys.stdout.flush()
    appendToFile(out2, out_file)
    start3 = time.time()
    out3 = shellGetOutput(s3)
    end3 = time.time()
    sys.stdout.write("Time 3: " + str(end3 - start3) + "\n")
    sys.stdout.flush()
    appendToFile(out3, out_file)
    start4 = time.time()
    out4 = shellGetOutput(s4)
    end4 = time.time()
    sys.stdout.write("Time 4: " + str(end4 - start4) + "\n")
    sys.stdout.flush()
    appendToFile(out4, out_file)
    start5 = time.time()
    out5 = shellGetOutput(s5)
    end5 = time.time()
    sys.stdout.write("Time 5: " + str(end5 - start5) + "\n")
    sys.stdout.flush()
    appendToFile(out5, out_file)

if __name__ == "__main__":
  main()
