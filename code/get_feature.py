import csv
import numpy as np
def MaxMinNormalization(x):
    if(np.max(x) - np.min(x)!=0):
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
#get HMM Profile
def readhmm(filename):
    with open(filename) as fs:
        hmm=[]
        data = csv.reader(fs)
        for line in data:
             line=[float(x) for x in line]
             hmm.append(line)
    hmm=np.array(hmm)
    return hmm
#get AAC
def gram1(hmm):
    feature1=np.zeros(20)
    for i in range(0,20):
        feature1[i]=hmm[:,i].mean()
    feature1=np.round(feature1,6)
    feature=[]
    for i in range(0,20):
        num=[]
        num.append(feature1[i])
        feature.extend(num)
    return feature
#get TPC
def gettpm(hmm):
    L=len(hmm)
    feature1=[]
    feature=[]
    for i in range(0,20):
        a=0
        for j in range(0,20):
            b=0
            for k in range(0,L-1):
                b=b+hmm[k][i]*hmm[k+1][j]
                a=a+hmm[k+1][j]*hmm[k][i]
            num=[]
            if(a!=0):
                num.append(b/a)
            else:
                num.append(0.0)
            feature1.extend(num)
    for i in range(0,400):
        num=[]
        num.append(feature1[i])
        feature.extend(num)
    return feature
def gettraindata():
    y_train0=[]
    y_train1=[]
    y_train=[]
    x_train=[]
    x_train0=[]
    for i in range(0,518):
        feature=[]
        hmm = readhmm('p' + str(i) + '.hhm.csv')
        feature.extend(gram1(hmm))
        feature.extend(gettpm(hmm))
        x_train.append(feature)
        y_train1.append(1.0)
    for i in range(0,550):
        feature=[]
        hmm = readhmm('n' + str(i) + '.hhm.csv')
        feature.extend(gram1(hmm))
        feature.extend(gettpm(hmm))
        x_train0.append(feature)
        y_train0.append(0.0)
    x_train.extend(x_train0)
    y_train1.extend(y_train0)
    y_train.extend(y_train1)
    return x_train,y_train
def gettestdata():
    y_test0 = []
    y_test1 = []
    y_test = []
    x_test = []
    x_test0 = []
    for i in range(0, 93):
        feature = []
        hmm = readhmm('186p' + str(i) + '.hhm.csv')
        feature.extend(gram1(hmm))
        feature.extend(gettpm(hmm))
        x_test.append(feature)
        y_test1.append(1.0)
    for i in range(0, 93):
        feature = []
        hmm = readhmm('186n' + str(i) + '.hhm.csv')
        feature.extend(gram1(hmm))
        feature.extend(gettpm(hmm))
        x_test0.append(feature)
        y_test0.append(0.0)
    x_test.extend(x_test0)
    y_test1.extend(y_test0)
    y_test.extend(y_test1)
    return x_test, y_test

#get feature
def getfeature():
    train_x,train_y=gettraindata()
    feature=[]
    for i in range(0,len(train_x)):
        feature.append(MaxMinNormalization(train_x[i]))
    csvFile = open("1075ACTP.csv", 'a',newline='')
    writer = csv.writer(csvFile,dialect='excel')
    for i in range(0,len(feature)):
        writer.writerow(feature[i])
    csvFile.close()

    test_x, test_y = gettestdata()
    feature = []
    for i in range(0, len(train_x)):
        feature.append(MaxMinNormalization(test_x[i]))
    csvFile = open("186ACTP.csv",'a',newline='')
    writer = csv.writer(csvFile,dialect='excel')
    for i in range(0, len(feature)):
        writer.writerow(feature[i])
    csvFile.close()

getfeature()