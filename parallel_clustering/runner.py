import os
import sys
import signal
import time
import subprocess
from run_exp_1 import run_1_1, run_1_2, run_1_3
from run_exp_2 import run_2_corr, run_2_mod
from run_exp_5 import run_5_corr, run_5_mod

def main():
  #run_1_1()
  #run_1_2()
  #run_1_3()
  run_5_corr()
  run_5_mod()

if __name__ == "__main__":
  main()
