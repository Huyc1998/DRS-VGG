import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import shutil
#Source files are from RRUFF official website
raw_files=glob.glob('D:\XXXX\XXXX/*.txt')
n=len(raw_files)
print(n)
fname = raw_files[np.random.randint(n)]
for fname in raw_files:
  basename, extension=os.path.splitext(os.path.basename(fname))
  compound, rlabel, ir_type, wv, n, orientation, data_status, rid = basename.split("__")
  
  path=os.path.join('raman_data', '{}/{}'.format(data_status, compound))
  os.makedirs(path, exist_ok=True)
  shutil.copy(fname, path)

