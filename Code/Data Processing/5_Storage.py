import numpy as np
import os 
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


data = pd.read_csv("D:\XXXX\XXXX/XXXX/56181601.csv",header=None,encoding="gbk")
Y=np.array(data)
Lab = pd.read_csv("D:\XXXX\XXXX/XXXX/56181601Label.csv",header=None,encoding="gbk")
Lab=np.array(Lab)
np.save('D:\XXXX\XXXX/XXXX/Data56181601.npy', Y)
np.save('D:\XXXX\XXXX/XXXX/Lab56181601.npy', Lab)






