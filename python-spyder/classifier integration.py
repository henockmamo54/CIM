# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:17:05 2020

@author: Henock
"""

from sklearn import datasets
from sklearn import metrics as m
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score 
import numpy as np
randomseed=5
np.random.seed(randomseed)


iris=datasets.load_iris()

x= iris.data
y=iris.target

xtest,xtrain,ytest,ytrain=train_test_split(x,y,random_state=randomseed,train_size=0.2)

rfc=RandomForestClassifier(n_estimators=100,random_state=randomseed)
rfc.fit(xtrain[:,0:1],ytrain)

ypred=rfc.predict(xtest[:,0:1])
ypredprob=rfc.predict_proba(xtest[:,0:1])

print('accuracy = ', m.accuracy_score(ytest,ypred))

#cvscore=cross_val_score(rfc, x[:,0:4], y, cv=10)
#print(cvscore)
#print('np.mean(cvscore)',np.mean(cvscore))
#print('np.mean(cvscore)',np.std(cvscore))

confmat=m.confusion_matrix(ytest,ypred)
confsumh=np.sum(confmat,axis=1)
print(confmat)
propconfmat=confmat.copy()

print('==============>>><<<')
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i]  
print(propconfmat)

# probalibity of class 0
pc0=[]




