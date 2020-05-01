# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:39:50 2020

@author: Henock
"""

import numpy as np
from sklearn import  datasets
from deslib.util import instance_hardness
import  pandas as pd
from sklearn import metrics as  m
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


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

# ===========================

data = pd.read_csv("../dataset/ionosphere.data" , header=None)
# data = pd.read_csv("../dataset/seeds_dataset.txt", sep="\t", header=None)
data = shuffle(data)
le = LabelEncoder()
data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
x = (data.iloc[:, :-1])
y = data.iloc[:, -1]
print(np.unique(y))

temp=pd.DataFrame(instance_hardness.kdn_score(np.array(x),np.array(y), 10)[0],columns=['val'])

# index_=temp[temp.val > 0.5].index #np.mean(temp.val)

# x=x.drop(index_)
# y=y.drop(index_)


xtrain,xtest,ytrain,ytest=train_test_split(np.array(x),np.array(y),random_state=randomseed,test_size=0.3)


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain,ytrain)

xgbpred=xgbc.predict(xtest)
print(m.f1_score(ytest,xgbpred,average='weighted'))
 


clf=[]
acc=[]
finalacc=[]
ypredproba_all=[]
ypredconfprob_all=[]

rf=RandomForestClassifier(random_state=randomseed, n_estimators=10)
rf.fit(xtrain,ytrain)
print('original score',m.f1_score(ytest,rf.predict(xtest),average='weighted'))



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
    # print(i, " ==================== ", comb[i])

    rf = RandomForestClassifier(random_state=randomseed, n_estimators=50)
    rf.fit(xtrain[:, comb[i]], ytrain)
    rfpred = rf.predict(xtest[:, comb[i]])
    # print(m.f1_score(ytest, rfpred,average='weighted'))

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
    # print(m.f1_score(ytest, xgbmodelpred,average='weighted'))

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

finalval=0
for i in range(len(acc)):
    finalval += weightvalga[i]*ypredproba_all[i]

print('f1_score',m.f1_score(ytest,np.argmax(finalval,axis=1),average='weighted'))
print('accuracy_score',m.accuracy_score(ytest,np.argmax(finalval,axis=1)))
    






































