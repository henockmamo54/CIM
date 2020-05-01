# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:14:22 2020

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
from sklearn.model_selection import  train_test_split
import warnings

warnings.filterwarnings('ignore')

data=datasets.load_iris()
x=data.data
y=data.target


temp=pd.DataFrame(instance_hardness.kdn_score(x, y, 12)[0],columns=['val'])

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=42,random_state=42)



rf=RandomForestClassifier(random_state=42, n_estimators=10)
rf.fit(xtrain,ytrain)
ypred=rf.predict(xtest)

vals=cross_val_score(rf,xtrain,np.array(ytrain),cv=10,verbose=0)
print(np.mean(vals),' std ',np.std(vals))

print('original score',m.f1_score(ytest,ypred,average='weighted'))
print(m.confusion_matrix(ytest,ypred))
print(m.classification_report(ytest,ypred))
print(rf.feature_importances_)





