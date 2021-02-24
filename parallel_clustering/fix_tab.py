import sys
import os
import time

def main():
  filename = sys.argv[1]
  idx = 0
  all_edges = []
  with open(filename) as fp:
    for line in fp:
      edges = [x.strip() for x in line.split()]
      all_edges.append(edges)
  output_fp = sys.argv[2]
  with open(output_fp, 'w') as fp:
    for edge in all_edges:
      print(edge[0] + "\t" + edge[1] + "\t" + edge[2],flush=True,file=fp)

if __name__ == "__main__":
  main()