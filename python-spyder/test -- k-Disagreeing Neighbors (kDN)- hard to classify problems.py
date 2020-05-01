# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:22:03 2020

@author: Henock
"""



# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:26:36 2020

@author: Henock
"""
"""
based on 
https://link.springer.com/article/10.1007/s10994-013-5422-z
An instance level analysis of data complexity
Michael R. Smith, Tony Martinez & Christophe Giraud-Carrier 
Machine Learning volume 95, pages225–256(2014)Cite this article
"""

import numpy as np
from sklearn import  datasets
from deslib.util import instance_hardness
import  pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics as  m
from sklearn.model_selection import cross_val_score
import warnings

randomseed=42
np.random.seed(randomseed)

import xgboost as xgb 

warnings.filterwarnings('ignore')


def swapcolumns(trainval, testval, coldindexval):
    trainval[trainval != coldindexval] = 5
    testval[testval != coldindexval] = 5

    trainval[trainval == coldindexval] = 0
    trainval[trainval == 5] = 1

    testval[testval == coldindexval] = 0
    testval[testval == 5] = 1

    return trainval, testval



# data=datasets.load_wine()

# x=data.data
# y=data.target


xtest=np.array(pd.read_csv('xtest.txt'))
xtrain=np.array(pd.read_csv('xtrain.txt'))
ytest=np.array(pd.read_csv('ytest.txt')).ravel()
ytrain=np.array(pd.read_csv('ytrain.txt')).ravel()

ytrain_original=ytrain.copy()
ytest_original = ytest.copy() 

x_temp=np.concatenate((xtrain,xtest))
y_temp=np.concatenate((ytrain,ytest))

# # deslib.util.instance_hardness.kdn_score(x, y, 10)
temp=pd.DataFrame(instance_hardness.kdn_score(x_temp, y_temp, 12)[0],columns=['val'])

# x_temp=np.column_stack((x_temp,temp))
# y_temp=np.column_stack((y_temp,temp))

# x_temp=pd.DataFrame(x_temp).drop(temp[temp.val > 0.39].index)
# y_temp=pd.DataFrame(y_temp).drop(temp[temp.val > 0.39].index )                  

# xtest=np.array(x_temp.iloc[-600:,:])
# xtrain=np.array(x_temp.iloc[:-600,:])
# ytest=np.array(y_temp.iloc[-600:,:])
# ytrain=np.array(y_temp.iloc[:-600,:])


#================================================= 

# Class 0
# ===========================
ytrain = ytrain_original.copy()
ytest = ytest_original.copy()
ytrain,ytest= swapcolumns(ytrain,ytest,2)
#=================================================


xtrain=np.column_stack((xtrain,temp.iloc[:-600,:]))
xtest=np.column_stack((xtest,temp.iloc[-600:,:]))
ytrain=np.column_stack((ytrain,temp.iloc[:-600,:]))
ytest=np.column_stack((ytest,temp.iloc[-600:,:]))


rf=RandomForestClassifier(random_state=42, n_estimators=10)
rf.fit(xtrain[:,:-1],ytrain[:,:-1])
rfypred=rf.predict(xtest[:,:-1])

# vals=cross_val_score(rf,xtrain[:,:-1],np.array(ytrain[:,:-1]),cv=10,verbose=0)
# print(np.mean(vals),' std ',np.std(vals))

print('rf original score',m.f1_score(ytest[:,:-1],rfypred,average='weighted'))
print(m.confusion_matrix(ytest[:,:-1],rfypred))
print(m.classification_report(ytest[:,:-1],rfypred))
print(rf.feature_importances_)


# lp=np.column_stack((ypred,ytest))
# lp=np.column_stack((lp,lp[:,2]>0.4))


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain[:,:-1],ytrain[:,:-1])

xgbpred=xgbc.predict(xtest[:,:-1])
print('xgbpred ',m.f1_score(ytest[:,:-1],xgbpred,average='weighted'))


result=pd.DataFrame(np.column_stack((ytest,rfypred,xgbpred)),columns=['actual','measure','rf','xgb'])

hardp=result[result.measure>0.39]
softp=result[result.measure<0.39]

print('rf hard score ',m.accuracy_score(hardp.actual,hardp.rf))
print('xgb hard score ',m.accuracy_score(hardp.actual,hardp.xgb))

print('rf soft score ',m.accuracy_score(softp.actual,softp.rf))
print('xgb soft score ',m.accuracy_score(softp.actual,softp.xgb))


























