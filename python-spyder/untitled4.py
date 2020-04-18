# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:09:01 2020

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

# def swapcolumns(trainval, testval, coldindexval):
#     trainval[trainval != coldindexval] = 5
#     testval[testval != coldindexval] = 5

#     trainval[trainval == coldindexval] = 0
#     trainval[trainval == 5] = 1

#     testval[testval == coldindexval] = 0
#     testval[testval == 5] = 1

#     return trainval, testval


randomseed=42
np.random.seed(randomseed)

data = pd.read_csv("../dataset/seeds_dataset.txt", sep="\t", header=None)
data = shuffle(data)
le = LabelEncoder()
data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
x = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
print(np.unique(y))  

clf=[]
acc=[]
finalacc=[]
ypredproba_all=[]
ypredconfprob_all=[]

xtrain,xtest,ytrain_original,ytest_original=train_test_split(x,y,random_state=randomseed,test_size=0.3) 
ytrain=ytrain_original.copy()
ytest=ytest_original.copy() 

rf=RandomForestClassifier(random_state=randomseed, n_estimators=10)
rf.fit(xtrain,ytrain)
print('original score',m.f1_score(ytest,rf.predict(xtest),average='weighted'))


# ====================************************====================
# ====================************************====================
# ====================************************====================
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
comb = list(itertools.combinations(np.arange(0, 7, 1), 4))

# generate 50 random numbers
randnums = []
for i in range(10):
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
    confsumh=np.sum(confmat,axis=1)
    propconfmat=confmat.copy()
    
    for i in range(propconfmat.shape[0]):
        propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
    ypredconfprob_all.append(propconfmat/100)

    xgbmodel = xgb.XGBClassifier(random_state=randomseed, n_estimators=50)
    xgbmodel.fit(xtrain, ytrain)
    xgbmodelpred = xgbmodel.predict(xtest)
    print(m.f1_score(ytest, xgbmodelpred,average='weighted'))

    clf.append(xgbmodel)
    acc.append(m.f1_score(ytest, xgbmodelpred,average='weighted'))
    ypredproba_all.append(xgbmodel.predict_proba(xtest))

    confmat = m.confusion_matrix(ytest, xgbmodelpred)
    confsumh=np.sum(confmat,axis=1)
    propconfmat=confmat.copy()
    
    for i in range(propconfmat.shape[0]):
        propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
    ypredconfprob_all.append(propconfmat/100)

# #=================================================
# #=================================================
# #=================================================
# #=================================================




import calculateWeightUsingGa2 as aresult
weightvalga=aresult.getbestvalues(acc)

finalval=0
for i in range(len(acc)):
    finalval += weightvalga[i]*ypredproba_all[i]

print('f1_score',m.f1_score(ytest,np.argmax(finalval,axis=1),average='weighted'))
print('accuracy_score',m.accuracy_score(ytest,np.argmax(finalval,axis=1)))




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


































