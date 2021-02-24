import os
import sys
import signal
import time
import subprocess
from run_exp_1 import rerun_1_2, rerun_1_3
from run_exp_2 import run_2_corr, run_2_mod
from run_exp_5 import run_5_corr, run_5_mod
from run_exp_6 import run_6_corr, run_6_mod
from run_exp_9 import run_9
from run_exp_7 import run_7
from run_exp_8 import run_8_corr, run_8_mod, run_8_corr_pr, run_8_mod_pr

def main():
  run_8_corr
  run_8_mod
  run_8_corr_pr
  run_8_mod_pr
  #run_1_3()
  #run_6_corr()
  #run_6_mod()
  #run_7()

if __name__ == "__main__":
  main()
