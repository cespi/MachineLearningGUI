import os
import numpy as np

def read_text_file(inputFile):
  fh = open(inputFile)
  X=np.loadtxt(fh,skiprows=1)
  return X
