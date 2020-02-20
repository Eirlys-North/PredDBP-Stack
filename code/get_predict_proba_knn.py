import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
import math
def proba():
    train_y=[]
    test_y=[]
    for i in range(0,518):
        train_y.append(1.0)
    for i in range(0,550):
        train_y.append(0.0)
    for i in range(0,93):
        test_y.append(1.0)
    for i in range(0,93):
        test_y.append(0.0)
    train_x = []
    with open("1075ACTP.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            row= [float(x) for x in line]
            train_x.append(row)
    test_x=[]
    with open("186ACTP.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            row = [float(x) for x in line]
            test_x.append(row)
    train_x=np.array(train_x)
    test_x=np.array(test_x)
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    clf = KNeighborsClassifier(n_neighbors=11)
    clf.fit(train_x,train_y)
    probas_y=clf.predict_proba(test_x)
    csvFile = open("186proba.csv",'a',newline='' )  # 创建csv文件
    writer = csv.writer(csvFile,dialect='excel')  # 创建写的对象
    writer.writerow(probas_y[:,1])
    csvFile.close()

    loo=LeaveOneOut()
    m = np.zeros((2, 2))
    tests_y = []
    probass_y = []
    for train_index, test_index in loo.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
        if (y_test == [1] and predicted == [1] ):
            cm = [[1, 0], [0, 0]]
        elif (y_test == [0] and predicted == [1]):
            cm = [[0, 0], [1, 0]]
        elif (y_test == [0] and predicted == [0]):
            cm = [[0, 0], [0, 1]]
        else:
            cm = [[0, 1], [0, 0]]
        m = m + cm
        probass_y.extend(clf.predict_proba(x_test)[:, 1])
        tests_y.extend(y_test)

    csvFile = open("1075proba.csv",'a',newline='')  # 创建csv文件
    writer = csv.writer(csvFile,dialect='excel')  # 创建写的对象
    writer.writerow(probass_y)
    csvFile.close()
    probass_y = np.array(probass_y)
    print(m)
    TP = m[0, 0]
    FN = m[0, 1]
    FP = m[1, 0]
    TN = m[1, 1]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (TP + FN)
    Specifity = TN / (TN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    print('OA:', accuracy)
    print('SN：', Sensitivity)
    print('SP:', Specifity)
    print('MCC:', MCC)
    print("AUC：", roc_auc_score(tests_y, probass_y))
proba()

