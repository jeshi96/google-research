import sys
import os
import time

def main():
  filename = sys.argv[1]
  idx = 0
  my_dict = dict()
  with open(filename) as fp:
    for line in fp:
      key = line.strip()
      if key in my_dict:
        my_dict[key].append(idx)
      else:
        my_dict[key] = [idx]
      idx += 1
  output_fp = sys.argv[2]
  with open(output_fp, 'w') as fp:
    for key in my_dict:
      for x in my_dict[key]:
        print(str(x),end='\t',flush=True,file=fp)
      print(flush=True,file=fp)

if __name__ == "__main__":
  main()