import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd 

Labels=np.array(pd.read_csv("D:\XXXX\XXXX/XXXX/56181601Label.csv",header=None,encoding="gbk"))
print(Labels)
Cou=Counter(Labels[:,-1])
Names=list(Cou.keys())
Names_Num=list(Cou.values())



x=Names_Num
a=[]
b=[]

s=set()
for i in x:
    if i not in s:
        s.add(i)
        print('Name:{}\tCount:{}'.format(i,x.count(i)))
        a.append(i)
        b.append(x.count(i))
print(a,b) 

plt.bar(x=a,height=b,width=0.75,label='1',color='steelblue',alpha=0.8)

plt.xlabel("Number of samples")
plt.ylabel("Amounts")
plt.title("Sample size distribution")
plt.show()

plt.plot(x)
plt.xlabel("%sClasses"%len(x))
plt.ylabel("Amounts")
plt.title("Number of samples in the original data set")


plt.legend()
plt.show()
