# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:29:30 2020

@author: Henock
"""

import  numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb 
from sklearn import metrics as m
from thundersvm import SVC as svmgpu
import calculateWeightUsingGa2 as aresult
import pandas as pd
import itertools
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

def swapcolumns(trainval, testval, coldindexval):
    trainval[trainval != coldindexval] = 5
    testval[testval != coldindexval] = 5

    trainval[trainval == coldindexval] = 0
    trainval[trainval == 5] = 1

    testval[testval == coldindexval] = 0
    testval[testval == 5] = 1

    return trainval, testval


randomseed=42
np.random.seed(randomseed)

xtest=np.array(pd.read_csv('xtest.txt'))
x=xtrain=np.array(pd.read_csv('xtrain.txt'))
ytest_original=np.array(pd.read_csv('ytest.txt')).ravel()
y=ytrain_original =np.array(pd.read_csv('ytrain.txt')).ravel()

# data=datasets.load_wine()
# x=data.data
# y=data.target

# data = pd.read_csv("../dataset/seeds_dataset.txt", sep="\t", header=None)
# data = shuffle(data)
# le = LabelEncoder()
# data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
# x = np.array(data.iloc[:, :-1])
# y = np.array(data.iloc[:, -1])
# print(np.unique(y))

# data = pd.read_csv("../dataset/ionosphere.data",  header=None)
# data = shuffle(data)
# le = LabelEncoder()
# data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
# x = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# print(np.unique(y))



original_scores=[]
trial1_scores=[]
trial2_scores=[]
trial3_scores=[]
trial4_scores=[]
trial5_scores=[]
trial6_scores=[]

from sklearn.model_selection import KFold 
kf = KFold(n_splits=10, random_state=randomseed, shuffle=True)
kf.get_n_splits(x)

print(kf)

data_cv=[]

for train_index, test_index in kf.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]    
    data_cv.append([[X_train, X_test],[y_train, y_test]])


for i in range(len(data_cv)):
    print('************************** ==> ',i)
    xtest=data_cv[i][0][1]
    xtrain=data_cv[i][0][0]
    ytest_original=data_cv[i][1][1]
    ytrain_original =data_cv[i][1][0] 
    
    
    # xtrain,xtest,ytrain_original,ytest_original=train_test_split(x,y,random_state=randomseed,test_size=0.3) 
    
    ytrain=ytrain_original.copy()
    ytest=ytest_original.copy() 
    
    # ytrain, ytest = swapcolumns(ytrain, ytest, 2)
    
    
    
    clf=[]
    acc=[]
    finalacc=[]
    ypredproba_all=[]
    ypredconfprob_all=[]
    
    rf=RandomForestClassifier(random_state=randomseed, n_estimators=10)
    rf.fit(xtrain,ytrain)
    print('original score',m.f1_score(ytest,rf.predict(xtest),average='weighted'))
    original_scores.append(m.accuracy_score(ytest,rf.predict(xtest)))
    
    #================================================= 
    
    # Class 0
    # ===========================
    ytrain = ytrain_original.copy()
    ytest = ytest_original.copy()
    ytrain,ytest= swapcolumns(ytrain,ytest,0)
    #=================================================
    
    # rf=RandomForestClassifier(random_state=randomseed, n_estimators=50)
    # rf.fit(xtrain,ytrain)
    # rfpred=rf.predict(xtest)
    # print(m.accuracy_score(ytest,rfpred)) 
    
    # clf.append(rf)
    # acc.append(m.accuracy_score(ytest,rfpred))
    # ypredproba_all.append(rf.predict_proba(xtest)) 
    
    # confmat=m.confusion_matrix(ytest,rfpred)
    # confsumh=np.sum(confmat,axis=0)
    # propconfmat=confmat.copy()
    # for i in range(propconfmat.shape[0]):
    #     propconfmat[:,i]= 100*propconfmat[:,i]/confsumh[i] 
    # ypredconfprob_all.append(propconfmat/100)
    
    ################
    xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
    xgbc.fit(xtrain,ytrain)
    
    xgbpred=xgbc.predict(xtest)
    print(m.f1_score(ytest,xgbpred,average='weighted'))
    
    
    clf.append(xgbc)
    acc.append(m.f1_score(ytest,xgbpred,average='weighted'))
    ypredproba_all.append(xgbc.predict_proba(xtest))
    
    confmat=m.confusion_matrix(ytest,xgbpred)
    confsumh=np.sum(confmat,axis=1)
    propconfmat=confmat.copy()
    for i in range(propconfmat.shape[0]):
        propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
    ypredconfprob_all.append(propconfmat/100)
    
    # =================================================
    # classs 1
    # =================================================
    ytrain = ytrain_original.copy()
    ytest = ytest_original.copy()
    ytrain, ytest = swapcolumns(ytrain, ytest, 1)
    # =================================================
    
    # rf = RandomForestClassifier(random_state=randomseed, n_estimators=50)
    # rf.fit(xtrain, ytrain)
    # rfpred = rf.predict(xtest)
    # print(m.accuracy_score(ytest, rfpred))
    
    # clf.append(rf)
    # acc.append(m.accuracy_score(ytest, rfpred))
    # ypredproba_all.append(rf.predict_proba(xtest))
    
    # confmat = m.confusion_matrix(ytest, rfpred)
    # confsumh = np.sum(confmat, axis=0)
    # propconfmat = confmat.copy()
    # for i in range(propconfmat.shape[0]):
    #     propconfmat[:, i] = 100 * propconfmat[:, i] / confsumh[i]
    # ypredconfprob_all.append(propconfmat / 100) 
    
    ################
    xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
    xgbc.fit(xtrain,ytrain)
    
    xgbpred=xgbc.predict(xtest)
    print(m.f1_score(ytest,xgbpred,average='weighted'))
    
    
    clf.append(xgbc)
    acc.append(m.f1_score(ytest,xgbpred,average='weighted'))
    ypredproba_all.append(xgbc.predict_proba(xtest))
    
    confmat=m.confusion_matrix(ytest,xgbpred)
    confsumh=np.sum(confmat,axis=1)
    propconfmat=confmat.copy()
    for i in range(propconfmat.shape[0]):
        propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
    ypredconfprob_all.append(propconfmat/100)
    
    #=================================================
    # classs 2
    #=================================================
    
    ytrain=ytrain_original.copy()
    ytest=ytest_original.copy()
    ytrain,ytest= swapcolumns(ytrain,ytest,2)
    
    #=================================================
    
    # rf=RandomForestClassifier(random_state=randomseed, n_estimators=50)
    # rf.fit(xtrain,ytrain)
    # rfpred=rf.predict(xtest)
    # print(m.accuracy_score(ytest,rfpred))
    
    # clf.append(rf)
    # acc.append(m.accuracy_score(ytest,rfpred))
    # ypredproba_all.append(rf.predict_proba(xtest))
    
    # confmat=m.confusion_matrix(ytest,rfpred)
    # confsumh=np.sum(confmat,axis=0)
    # propconfmat=confmat.copy()
    # for i in range(propconfmat.shape[0]):
    #     propconfmat[:,i]= 100*propconfmat[:,i]/confsumh[i] 
    # ypredconfprob_all.append(propconfmat/100)
    
    ################
    xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
    xgbc.fit(xtrain,ytrain)
    
    xgbpred=xgbc.predict(xtest)
    print(m.f1_score(ytest,xgbpred,average='weighted'))
    
    
    clf.append(xgbc)
    acc.append(m.f1_score(ytest,xgbpred,average='weighted'))
    ypredproba_all.append(xgbc.predict_proba(xtest))
    
    confmat=m.confusion_matrix(ytest,xgbpred)
    confsumh=np.sum(confmat,axis=1)
    propconfmat=confmat.copy()
    for i in range(propconfmat.shape[0]):
        propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
    ypredconfprob_all.append(propconfmat/100)
    
    
    # #=================================================
    
    
    import calculateWeightUsingGa2 as aresult
    weightvalga=aresult.getbestvalues(acc)
    
    finalcol = np.zeros((ytest.shape[0], 3))
    finalcol[:, 0] = ypredproba_all[0][:, 0]*weightvalga[0]
    finalcol[:, 1] = ypredproba_all[1][:, 0]*weightvalga[1]
    finalcol[:, 2] = ypredproba_all[2][:, 0]*weightvalga[2]
    
    finalpred = np.argmax(finalcol, axis=1)
    
    ytest = ytest_original.copy()
    print('accuracy_score',m.accuracy_score(ytest, finalpred))
    # accuracy_score 0.6433333333333333
    
    trial1_scores.append(m.accuracy_score(ytest, finalpred))
    
    # #=================================================
    
    import calculateWeightUsingGa2 as aresult
    weightvalga=aresult.getbestvalues([
        ypredconfprob_all[0][0][0] * ypredconfprob_all[1][1][1]*ypredconfprob_all[2][1][1],
        ypredconfprob_all[1][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[2][1][1],
        ypredconfprob_all[2][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[1][1][1]])
    
    finalcol = np.zeros((ytest.shape[0], 3))
    finalcol[:, 0] = ypredproba_all[0][:, 0]*weightvalga[0]
    finalcol[:, 1] = ypredproba_all[1][:, 0]*weightvalga[1]
    finalcol[:, 2] = ypredproba_all[2][:, 0]*weightvalga[2]
    
    finalpred = np.argmax(finalcol, axis=1)
    
    ytest = ytest_original.copy()
    print('accuracy_score',m.accuracy_score(ytest, finalpred))
    # accuracy_score 0.6783333333333333
    # accuracy_score 0.715
    # accuracy_score 0.7216666666666667
    trial2_scores.append(m.accuracy_score(ytest, finalpred))
    
    # #=================================================
    
    import calculateWeightUsingGa2 as aresult
    weightvalga=aresult.getbestvalues([
        ypredconfprob_all[0][0][0] * ypredconfprob_all[1][1][1]*ypredconfprob_all[2][1][1],
        ypredconfprob_all[1][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[2][1][1],
        ypredconfprob_all[2][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[1][1][1]])
    
    finalcol = np.zeros((ytest.shape[0], 3))
    finalcol[:, 0] = ypredproba_all[0][:, 0] * weightvalga[0] + ypredproba_all[0][:, 1] * ypredconfprob_all[0][0][1]
    finalcol[:, 1] = ypredproba_all[1][:, 0] * weightvalga[1] + ypredproba_all[1][:, 1] * ypredconfprob_all[1][0][1]
    finalcol[:, 2] = ypredproba_all[2][:, 0] * weightvalga[2] + ypredproba_all[2][:, 1] * ypredconfprob_all[2][0][1]
    
    finalpred = np.argmax(finalcol, axis=1)
    
    ytest = ytest_original.copy()
    print('accuracy_score',m.accuracy_score(ytest, finalpred))
    trial3_scores.append(m.accuracy_score(ytest, finalpred))
    
    # #=================================================
    
    import calculateWeightUsingGa2 as aresult
    weightvalga=aresult.getbestvalues([
        ypredconfprob_all[0][0][0] * ypredconfprob_all[1][1][1]*ypredconfprob_all[2][1][1] ,
        -ypredconfprob_all[0][0][1],
        
        ypredconfprob_all[1][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[2][1][1],
        -ypredconfprob_all[1][0][1],
        
        ypredconfprob_all[2][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[1][1][1],
        -ypredconfprob_all[2][0][1],
        
        ])
    
    finalcol = np.zeros((ytest.shape[0], 3))
    finalcol[:, 0] = ypredproba_all[0][:, 0]*weightvalga[0] + ypredproba_all[0][:, 1]*weightvalga[1]
    finalcol[:, 1] = ypredproba_all[1][:, 0]*weightvalga[2] + ypredproba_all[1][:, 1]*weightvalga[3]
    finalcol[:, 2] = ypredproba_all[2][:, 0]*weightvalga[4] + ypredproba_all[2][:, 1]*weightvalga[5]
    
    finalpred = np.argmax(finalcol, axis=1)
    
    ytest = ytest_original.copy()
    print('accuracy_score',m.accuracy_score(ytest, finalpred))
    
    trial4_scores.append(m.accuracy_score(ytest, finalpred))
    
    # #=================================================

    import calculateWeightUsingGa2 as aresult
    weightvalga=aresult.getbestvalues([
        ypredconfprob_all[0][0][0] ,
        ypredconfprob_all[1][0][0] ,
        ypredconfprob_all[2][0][0] ])
    
    finalcol = np.zeros((ytest.shape[0], 3))
    finalcol[:, 0] = weightvalga[0] * (ypredproba_all[0][:, 0] * ypredconfprob_all[0][0][0] + ypredproba_all[0][:, 1] * ypredconfprob_all[0][0][1])
    finalcol[:, 1] = weightvalga[1] * (ypredproba_all[1][:, 0] * ypredconfprob_all[1][0][0] + ypredproba_all[1][:, 1] * ypredconfprob_all[1][0][1])
    finalcol[:, 2] = weightvalga[2] * (ypredproba_all[2][:, 0] * ypredconfprob_all[2][0][0] + ypredproba_all[2][:, 1] * ypredconfprob_all[2][0][1])
    
    finalpred = np.argmax(finalcol, axis=1)
    
    ytest = ytest_original.copy()
    print('accuracy_score',m.accuracy_score(ytest, finalpred))
    trial5_scores.append(m.accuracy_score(ytest, finalpred))
    
    # #=================================================
    
    import calculateWeightUsingGa2 as aresult
    weightvalga=aresult.getbestvalues([
        ypredconfprob_all[0][0][0] ,
        ypredconfprob_all[1][0][0] ,
        ypredconfprob_all[2][0][0] ])
    
    finalcol = np.zeros((ytest.shape[0], 3))
    finalcol[:, 0] =  (weightvalga[0] *ypredproba_all[0][:, 0] * ypredconfprob_all[0][0][0] + ypredproba_all[0][:, 1] * ypredconfprob_all[0][0][1])
    finalcol[:, 1] = (weightvalga[1] * ypredproba_all[1][:, 0] * ypredconfprob_all[1][0][0] + ypredproba_all[1][:, 1] * ypredconfprob_all[1][0][1])
    finalcol[:, 2] =  (weightvalga[2] *ypredproba_all[2][:, 0] * ypredconfprob_all[2][0][0] + ypredproba_all[2][:, 1] * ypredconfprob_all[2][0][1])
    
    finalpred = np.argmax(finalcol, axis=1)
    
    ytest = ytest_original.copy()
    print('accuracy_score',m.accuracy_score(ytest, finalpred))
    trial6_scores.append(m.accuracy_score(ytest, finalpred))
    
    # #=================================================

    
    
    # # _rf_p=weightvalga[0]*ypredproba_all[0]
    # # _svm_p=weightvalga[1]*ypredproba_all[1]
    # # _xgb_p=weightvalga[2]*ypredproba_all[2]
    # # _temp=np.argmax(_rf_p+_svm_p+_xgb_p,axis=1)
    # # print(m.f1_score(ytest,_temp,average='weighted'))
    
    # # finalval=0
    # # for i in range(len(acc)):
    # #     finalval += weightvalga[i]*ypredproba_all[i]
    
    # # ytest = ytest_original.copy()
    
    # # print('f1_score',m.f1_score(ytest,np.argmax(finalval,axis=1),average='weighted'))
    # # print('accuracy_score',m.accuracy_score(ytest,np.argmax(finalval,axis=1)))
        
    
print('original_scores',np.mean(original_scores),np.std(original_scores))
print('trial1_scores',np.mean(trial1_scores),np.std(trial1_scores))
print('trial2_scores',np.mean(trial2_scores),np.std(trial2_scores))
print('trial3_scores',np.mean(trial3_scores),np.std(trial3_scores))
print('trial2_scores',np.mean(trial4_scores),np.std(trial4_scores))
print('trial3_scores',np.mean(trial5_scores),np.std(trial5_scores))






