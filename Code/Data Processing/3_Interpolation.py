import glob
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import interpolate as spi
import csv

raw_files = glob.glob('D:\XXXX\XXXX/XXXX\*/*.txt')
n = len(raw_files)

new_x = np.linspace(50, 1650, num=1601)  
rruff = pd.DataFrame(np.zeros((n,1601)))
x_n = pd.DataFrame(np.zeros((n,1601)))

f = open('D:\XXXX\XXXX/XXXX/56181601.csv','w',encoding='utf-8',newline='')# Create a CSV file for the process
csv_writer = csv.writer(f)
f3 = open('D:\XXXX\XXXX/XXXX/56181601Label.csv','w',encoding='utf-8',newline='')
csv_writer3=csv.writer(f3)


def MaxMinNormalization(x):
     x = (x-np.min(x)) / (np.max(x)-np.min(x))
     return x

def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

for fname,i in zip(raw_files,range(int(n))):
    basename, extension = os.path.splitext(os.path.basename(fname))
    compound, rlabel, ir_type, wv, n, orientation, data_status, rid = basename.split("__")
    #path=os.path.join('raman_data', '{}/{}'.format(data_status, compound))
    #os.makedirs(path, exist_ok=True)
    #shutil.copy(fname, path)

    data = np.genfromtxt(fname,
                         dtype=None,
                         comments='##',  
                         delimiter=',',
                         names=['Raman_shift','Intensity'],
                         encoding = "utf-8")
    #print('{} of {} files'.format(fname, len(raw_files)))  
    
    Rs=data['Raman_shift'];In=data['Intensity']
    

    # In = MaxMinNormalization(In)
    fi = spi.interp1d(Rs, In, kind = 'linear', bounds_error=False, fill_value=0)
    ynew = fi(new_x) 
    # plt.plot(new_x, ynew)
    # plt.show()
    ynew=np.array(ynew)
    ynew=np.squeeze(ynew)


    ynew = MaxMinNormalization(ynew) 
    # plt.plot(new_x,ynew)
    # plt.show()
    csv_writer.writerow(ynew)
    csv_writer3.writerow([compound])
f.close()
f3.close()