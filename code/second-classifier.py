import csv
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
#knn-lr-dt-svm(linear)-svm(rbf)-rf-xgb
def gettest():
    proba = []
    with open("186proba.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            row = [float(x) for x in line]
            proba.append(row)
    x_test = []
    xtest = []
    x_test.append(proba[0])
    x_test.append(proba[1])
    x_test.append(proba[2])
    #x_test.append(proba[3])
    #x_test.append(proba[4])
    #x_test.append(proba[5])
    x_test.append(proba[6])
    x_test = np.array(x_test)
    for j in range(0, x_test.shape[1]):
        xtest.append(x_test[:, j])
    ytest = []
    for i in range(0, 93):
        ytest.append(1)
    for i in range(0, 93):
        ytest.append(0)
    return xtest,ytest
def gettrain():
    proba = []
    with open("1075proba.csv") as fs:
        data = csv.reader(fs)
        for line in data:
            row = [float(x) for x in line]
            proba.append(row)
    x_train = []
    xtrain = []
    x_train.append(proba[0])
    x_train.append(proba[1])
    x_train.append(proba[2])
    #x_train.append(proba[3])
    #x_train.append(proba[4])
    #x_train.append(proba[5])
    x_train.append(proba[6])
    x_train= np.array(x_train)
    for j in range(0, x_train.shape[1]):
        xtrain.append(x_train[:, j])
    ytrain = []
    for i in range(0, 518):
        ytrain.append(1)
    for i in range(0, 550):
        ytrain.append(0)
    return xtrain, ytrain

def test():
    xtrain,ytrain=gettrain()
    xtest,ytest=gettest()
    xtrain=np.array(xtrain)
    xtest=np.array(xtest)
    ytrain=np.array(ytrain)
    ytest=np.array(ytest)
    '''
    # scaler = MinMaxScaler()
    # scaler = StandardScaler()
    #scaler = RobustScaler()
    # scaler=Normalizer()
    #xtrain= scaler.fit_transform(xtrain)
    param_test1 = {'n_estimators': range(100, 500, 10)}
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                 min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                                 subsample=0.8, random_state=10),
                            param_grid=param_test1, scoring='accuracy', iid=False, cv=10)
    gsearch1.fit(xtrain, ytrain)
    print(gsearch1.best_params_, gsearch1.best_score_)
    '''
    '''
    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
    gsearch2 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=460, min_samples_leaf=20,
                                            max_features='sqrt', subsample=0.8, random_state=10),
        param_grid=param_test2, scoring='accuracy', iid=False, cv=10)
    gsearch2.fit(xtrain, ytrain)
    print(gsearch2.best_params_, gsearch2.best_score_)
    '''
    '''
    # scaler = MinMaxScaler()
    # scaler = StandardScaler()
    #scaler = RobustScaler()
    # scaler=Normalizer()
    #xtrain= scaler.fit_transform(xtrain)
    #xtest = scaler.fit_transform(xtest)
    for i in range(1,100,1):
        gsearch3 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=18, max_depth=2,min_samples_split=100,
                                        min_samples_leaf=i, subsample=0.5, random_state=10)
        gsearch3.fit(xtrain,ytrain)
        predicted=gsearch3.predict(xtest)
        accuracy=accuracy_score(ytest,predicted)
        print(i,accuracy)
    '''


def independenttest():
    train_x, train_y = gettrain()
    test_x, test_y = gettest()
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    # scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = RobustScaler()
    #scaler=Normalizer()
    # train_x=scaler.fit_transform(train_x)
    #test_x=scaler.fit_transform(test_x)
    clf=GradientBoostingClassifier(learning_rate=0.01, n_estimators=420, subsample=0.5,
                                 random_state=10, min_samples_split=100)
    clf.fit(train_x, train_y)
    predict = clf.predict(test_x)
    accracy = accuracy_score(test_y, predict)
    print(accracy)
    matrix = confusion_matrix(test_y, predict)
    print(matrix)
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TP = matrix[1, 1]
    Sensitivity = TP / (TP + FN)
    Specifity = TN / (TN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    print('SN：', Sensitivity)
    print('SP:', Specifity)
    print('MCC:', MCC)
    probas_y = clf.predict_proba(test_x)
    print("AUC：", roc_auc_score(test_y, probas_y[:, 1]))
    fpr, tpr, thresholds = roc_curve(test_y, probas_y[:, 1])
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    file = open('SM186prob.csv', 'a', newline='')
    content = csv.writer(file, dialect='excel')
    content.writerow(probas_y[:, 1])
def jackknife():
    train_x, train_y = gettrain()
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler=Normalizer()
    # train_x=scaler.fit_transform(train_x)
    clf=GradientBoostingClassifier(learning_rate=0.1, n_estimators=470, min_samples_leaf=20,
                                            max_features='sqrt', subsample=0.8, random_state=10,
                                   max_depth=9,min_samples_split=300)
    loo=LeaveOneOut()
    m = np.zeros((2, 2))
    tests_y = []
    probass_y = []
    for train_index, test_index in loo.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
        if (predicted == [1] and y_test == [1]):
            cm = [[1, 0], [0, 0]]
        elif (predicted == [1] and y_test == [0]):
            cm = [[0, 0], [1, 0]]
        elif (predicted == [0] and y_test == [0]):
            cm = [[0, 0], [0, 1]]
        else:
            cm = [[0, 1], [0, 0]]
        m = m + cm
        probass_y.extend(clf.predict_proba(x_test)[:, 1])
        tests_y.extend(y_test)
    file = open('SM1075prob.csv', 'a', newline='')  # 打开文件
    content = csv.writer(file, dialect='excel')  # 设定文件写入模式
    content.writerow(probass_y)
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
    print('MCC', MCC)
    print("AUC：", roc_auc_score(tests_y, probass_y))
    fpr, tpr, thresholds = roc_curve(tests_y, probass_y)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
#test()
independenttest()
jackknife()