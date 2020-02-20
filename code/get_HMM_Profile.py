
import os
import numpy as np
import csv
import math
def readhmm(path,filedir,filename):
    L=0
    hmm=[]
    if(filedir=='binder'):
        str='186p'+filename+'.csv'
    else:
        str='186n'+filename+'.csv'
    file=open(str,'a',newline='')
    content=csv.writer(file,dialect='excel')
    fr=open(path+'/'+filename)
    arryOlines=fr.readlines()
    filelen=len(arryOlines)
    for i in range(0,filelen):
        str=arryOlines[i]
        if(str.strip()=='#'):
            break
        L=L+1
    L=L+5
    for i in range(L,filelen-3,3):
       strhmm=arryOlines[i].strip()
       strhmm=strhmm+arryOlines[i+1].strip()
       strhmm=strhmm+arryOlines[i+1].strip()
       strhmm=strhmm.split()
       num=strhmm[2:22]
       #convert to probability values
       for j in range(0,20):
           if(num[j]=='*'):
               num[j]=0
       num=[float(x) for x in num]
       for j in range(0,20):
           if(num[j]!=0):
               num[j]=math.pow(2,(-num[j])/1000)
               num[j]=round(num[j],6)
       hmm.append(num)
       content.writerow(num)
    hmm=np.array(hmm)
    return hmm

for i in range(0,93):
    str1=str(i)
    hmm=readhmm('C:/Users/mrswang/Desktop/Newphmm_1075/186/binder','binder',str1+'.hhm')
for j in range(0,93):
    str1=str(j)
    hmm=readhmm('C:/Users/mrswang/Desktop/Newphmm_1075/186/nobinder','nobinder',str1+'.hhm')