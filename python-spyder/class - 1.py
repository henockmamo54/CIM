# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:16:40 2020

@author: Henock
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:56:12 2020

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
xtrain=np.array(pd.read_csv('xtrain.txt'))
ytest_original=np.array(pd.read_csv('ytest.txt')).ravel()
ytrain_original =np.array(pd.read_csv('ytrain.txt')).ravel()

# data=datasets.load_wine()
# x=data.data
# y=data.target

# data = pd.read_csv("../dataset/seeds_dataset.txt", sep="\t", header=None)
# data = shuffle(data)
# le = LabelEncoder()
# data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
# x = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# print(np.unique(y))

# data = pd.read_csv("../dataset/ionosphere.data",  header=None)
# data = shuffle(data)
# le = LabelEncoder()
# data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
# x = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# print(np.unique(y))




# xtrain,xtest,ytrain_original,ytest_original=train_test_split(x,y,random_state=randomseed,test_size=0.3) 

ytrain=ytrain_original.copy()
ytest=ytest_original.copy() 

# ytrain, ytest = swapcolumns(ytrain, ytest, 2)



clf=[]
acc=[]
finalacc=[]
ypredproba_all=[]
ypredconfprob_all=[]


#================================================= 

# Class 0
# ===========================
ytrain = ytrain_original.copy()
ytest = ytest_original.copy()
ytrain,ytest= swapcolumns(ytrain,ytest,1)
#=================================================

rf=RandomForestClassifier(random_state=randomseed, n_estimators=10)
rf.fit(xtrain,ytrain)
print('original score',m.f1_score(ytest,rf.predict(xtest),average='weighted'))


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain,ytrain)
xgbpred=xgbc.predict(xtest)
print('original score 2',m.f1_score(ytest,xgbpred,average='weighted'))


#================================================= 

rf=RandomForestClassifier(random_state=randomseed, n_estimators=10)
rf.fit(xtrain,ytrain)
rfpred=rf.predict(xtest)
print(m.f1_score(ytest,rfpred,average='weighted'))

clf.append(rf)
acc.append(m.f1_score(ytest,rfpred,average='weighted'))
ypredproba_all.append(rf.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,rfpred)
confsumh=np.sum(confmat,axis=1)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)

#=================================================
svc=svmgpu(random_state=randomseed,probability=True,C=100,gamma=0.0001)
svc.fit(xtrain,ytrain)

svcpred=svc.predict(xtest)
print(m.f1_score(ytest,svcpred,average='weighted'))

clf.append(svc)
acc.append(m.f1_score(ytest,svcpred,average='weighted'))
ypredproba_all.append(svc.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,svcpred)
confsumh=np.sum(confmat,axis=1)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)

#=================================================
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
#=================================================
# generate combinations of features 12,6
comb = list(itertools.combinations(np.arange(0, 12, 1), 7))

# generate 50 random numbers
randnums = []
for i in range(50):
    randnums.append(random.randrange(0, len(comb)))

print(randnums)

comb = np.array(comb)[randnums, :]


for i in range(len(comb)):
    print(i, " ==================== ", comb[i])

    rf = RandomForestClassifier(random_state=randomseed, n_estimators=50)
    rf.fit(xtrain[:, comb[i]], ytrain)
    rfpred = rf.predict(xtest[:, comb[i]])
    print(m.f1_score(ytest, rfpred,average='weighted'))

    clf.append(rf)
    acc.append(m.f1_score(ytest, rfpred,average='weighted'))
    ypredproba_all.append(rf.predict_proba(xtest[:, comb[i]]))

    confmat = m.confusion_matrix(ytest, rfpred)
    confsumh = np.sum(confmat, axis=0)
    propconfmat = confmat.copy()
    for i in range(propconfmat.shape[0]):
        propconfmat[:, i] = 100 * propconfmat[:, i] / confsumh[i]
    ypredconfprob_all.append(propconfmat / 100)

    xgbmodel = xgb.XGBClassifier(random_state=randomseed, n_estimators=50)
    xgbmodel.fit(xtrain, ytrain)
    xgbmodelpred = xgbmodel.predict(xtest)
    print(m.f1_score(ytest, xgbmodelpred,average='weighted'))

    clf.append(xgbmodel)
    acc.append(m.f1_score(ytest, xgbmodelpred,average='weighted'))
    ypredproba_all.append(xgbmodel.predict_proba(xtest))

    confmat = m.confusion_matrix(ytest, xgbmodelpred)
    confsumh = np.sum(confmat, axis=0)
    propconfmat = confmat.copy()
    for i in range(propconfmat.shape[0]):
        propconfmat[:, i] = 100 * propconfmat[:, i] / confsumh[i]
    ypredconfprob_all.append(propconfmat / 100)

# #=================================================


import calculateWeightUsingGa2 as aresult
weightvalga=aresult.getbestvalues(acc)
# _rf_p=weightvalga[0]*ypredproba_all[0]
# _svm_p=weightvalga[1]*ypredproba_all[1]
# _xgb_p=weightvalga[2]*ypredproba_all[2]
# _temp=np.argmax(_rf_p+_svm_p+_xgb_p,axis=1)
# print(m.f1_score(ytest,_temp,average='weighted'))

finalval=0
for i in range(len(acc)):
    finalval += weightvalga[i]*ypredproba_all[i]

print('f1_score',m.f1_score(ytest,np.argmax(finalval,axis=1),average='weighted'))
print('accuracy_score',m.accuracy_score(ytest,np.argmax(finalval,axis=1)))
    




