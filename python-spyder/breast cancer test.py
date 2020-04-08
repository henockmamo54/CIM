# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:31:45 2020

@author: Henock
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as m
import numpy as np
randomseed=5
np.random.seed(randomseed)


data=datasets.load_breast_cancer()

x=data.data
y=data.target

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=randomseed)

rf=RandomForestClassifier(random_state=randomseed,n_estimators=100)
rf.fit(xtrain,ytrain)

ypred=rf.predict(xtest)

print(m.accuracy_score(ytest,ypred))