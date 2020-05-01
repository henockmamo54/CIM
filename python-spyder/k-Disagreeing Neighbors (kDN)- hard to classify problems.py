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

warnings.filterwarnings('ignore')

data=datasets.load_wine()

# x=data.data
# y=data.target


xtest=np.array(pd.read_csv('xtest.txt'))
xtrain=np.array(pd.read_csv('xtrain.txt'))
ytest=np.array(pd.read_csv('ytest.txt')).ravel()
ytrain=np.array(pd.read_csv('ytrain.txt')).ravel()

x_temp=np.concatenate((xtrain,xtest))
y_temp=np.concatenate((ytrain,ytest))


# deslib.util.instance_hardness.kdn_score(x, y, 10)

temp=pd.DataFrame(instance_hardness.kdn_score(x_temp, y_temp, 12)[0],columns=['val'])


x_temp=pd.DataFrame(x_temp).drop(temp[temp.val < 0.5].index)
y_temp=pd.DataFrame(y_temp).drop(temp[temp.val < 0.5].index )                  

xtest=x_temp.iloc[-600:,:]
xtrain=x_temp.iloc[:-600,:]
ytest=y_temp.iloc[-600:,:]
ytrain=y_temp.iloc[:-600,:]



# print(
# temp[0][temp[0]>0.5]
# )




rf=RandomForestClassifier(random_state=42, n_estimators=10)
rf.fit(xtrain,ytrain)
ypred=rf.predict(xtest)

vals=cross_val_score(rf,xtrain,np.array(ytrain),cv=10,verbose=0)
print(np.mean(vals),' std ',np.std(vals))

print('original score',m.f1_score(ytest,ypred,average='weighted'))
print(m.confusion_matrix(ytest,ypred))
print(m.classification_report(ytest,ypred))
print(rf.feature_importances_)



