
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

count1 = 0   
for filename in os.listdir('D:\XXXX\XXXX/XXXX'):#From step 1
    #print (filename)
    count1 += 1
print ('There are %s labels in total'%count1)
Label_Num = []

s=0
j = 0          
k=0
for filename in os.listdir('D:\XXXX\XXXX/XXXX'):
    #print (filename+':'),       
    count = 0        
    path = os.path.join('D:\XXXX\XXXX/XXXX', filename)
    ls = os.listdir(path)
    for i in ls:
        if os.path.isfile(os.path.join(path,i)):#
            count += 1
    Label_Num.append(count)
    s+=count
    if count > 50:
        j += 1
        print (filename,count),
    if count < 6:
        k += 1
    #     print (filename,count),

print ('There are %s labels over the limit'%j)
print ('There are %s labels below the limit'%k)
print('The total number of samples is：%s'%s)
print(Label_Num)
plt.plot(sorted(Label_Num,reverse = True))
plt.xlabel("%sClasses"%len(Label_Num))
plt.ylabel("Amounts")
plt.title("Number of samples in the original data set")
#plt.legend()
plt.show()

#Deletion-----------------(Another complete program)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

count1 = 0    
for filename in os.listdir('D:\XXXX\XXXX/XXXX'):
    #print (filename)
    count1 += 1
print ('There are %s labels in total'%count1)
Label_Num = []
s=0
j = 0          
k=0
a=0
for filename in os.listdir('D:\XXXX\XXXX/XXXX'):
    #print (filename+':'),    
    count = 0     
    path = os.path.join('D:\XXXX\XXXX/XXXX', filename)
    ls = os.listdir(path)
    for i in ls:
        if os.path.isfile(os.path.join(path,i)):
            count += 1
    Label_Num.append(count)
    s+=count

    if count > 50:
        j += 1
        print (filename,count),
        print (path)
        shutil.rmtree(path)  
    if count <6:
        k += 1
        print (filename,count),
        print (path)
        shutil.rmtree(path)
print ('There are %s labels over the limit'%j)
print ('There are %s labels below the limit'%k)
print('The final number of samples is：%s'%s)