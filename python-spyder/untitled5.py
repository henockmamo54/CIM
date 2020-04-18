# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:04:02 2020

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


randomseed=42
np.random.seed(randomseed)


def swapcolumns(trainval, testval, coldindexval):
    trainval[trainval != coldindexval] = 5
    testval[testval != coldindexval] = 5

    trainval[trainval == coldindexval] = 0
    trainval[trainval == 5] = 1

    testval[testval == coldindexval] = 0
    testval[testval == 5] = 1

    return trainval, testval


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


ytrain,ytest= swapcolumns(ytrain,ytest,0)

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

one=[]
two=[] 
cone=[]
ctwo=[]

ypredconfprob_all=np.array(ypredconfprob_all)
    

# for i in range(2):
    
#     for j in range(3):
#         for h in range(3):
#             for k in range(3):
                
#                 if(i==0):
#                     one.append([j,h,k])
#                 elif(i==1):
#                     two.append([j,h,k])


temp=np.array([[0.9,0.1],[0.1,0.9]])

for i in range(2):
    
    for j in range(2):
        for h in range(2):
            for k in range(2):
                
                if(i==0):
                    one.append([j,h,k])
                    cone.append(temp[i][j]*temp[i][h]*temp[i][k])
                elif(i==1):
                    two.append([j,h,k])
                    ctwo.append(temp[i][j]*temp[i][h]*temp[i][k])







































