# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:31:25 2020

@author: Henock
"""

import numpy as np
from sklearn import  datasets
from deslib.util import instance_hardness
import  pandas as pd
from sklearn import metrics as  m
from sklearn.model_selection import cross_val_score
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
import warnings
warnings.filterwarnings('ignore')

randomseed=42
np.random.seed(randomseed)



# ===========================

def computePercentageOfCorrectHardToCProblems(xtest,ytest,ypred): 
    var_temp=pd.DataFrame(xtest)
    hd_ind=var_temp[var_temp.iloc[:,-1]>np.mean(temp.val)].index
    
    correctCount=0
    for i in hd_ind:
        if(ytest[i]==ypred[i]):
            correctCount=correctCount+1
            
    return (correctCount/len(hd_ind))
    

# ===========================

# data = pd.read_csv("../dataset/ionosphere.data" , header=None)
data = pd.read_csv("../dataset/seeds_dataset.txt", sep="\t", header=None)
data = shuffle(data)
le = LabelEncoder()
data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
x = (data.iloc[:, :-1])
y = data.iloc[:, -1]
print(np.unique(y))

temp=pd.DataFrame(instance_hardness.kdn_score(np.array(x),np.array(y), 10)[0],columns=['val'])

x=np.column_stack((x,temp))


# index_=temp[temp.val > 0.5].index #np.mean(temp.val)
# x=x.drop(index_)
# y=y.drop(index_)


xtrain,xtest,ytrain,ytest=train_test_split(np.array(x),np.array(y),random_state=randomseed,test_size=0.3)


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain,ytrain)
xgbpred=xgbc.predict(xtest)
print(m.f1_score(ytest,xgbpred,average='weighted'))

# ==========================================================================
# generate bag of data

sample_size= int(.6*xtrain.shape[0])
bags=[]
no_bags = 500 # generate random 50 bags of dataset with size = sample size 

for i in range(no_bags):
    indexs=[]
    for i in range(sample_size):
        indexs.append(random.randint(0,sample_size))
    
    bags.append([xtrain[indexs,:],ytrain[indexs]]) 
    
# ==========================================================================

clf=[]
acc=[]
hd_clf=[]
finalacc=[]
ypredproba_all=[]
ypredconfprob_all=[]

for i in range(no_bags):
    
    _xtrain=bags[i][0]
    _ytrain=bags[i][1]
    
    xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=50)
    xgbc.fit(_xtrain,_ytrain)    
    xgbpred=xgbc.predict(xtest)
    print(i)
    # print(i,' f1_score ',m.f1_score(ytest,xgbpred,average='weighted'),
    #       computePercentageOfCorrectHardToCProblems(xtest,ytest,xgbpred))
    
    clf.append(xgbc)
    acc.append(m.f1_score(ytest,xgbpred,average='weighted'))
    hd_clf.append(computePercentageOfCorrectHardToCProblems(xtest,ytest,xgbpred))
    ypredproba_all.append(xgbc.predict_proba(xtest))
    
    confmat=m.confusion_matrix(ytest,xgbpred)
    confsumh=np.sum(confmat,axis=1)
    propconfmat=confmat.copy()
    for i in range(propconfmat.shape[0]):
        propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
    ypredconfprob_all.append(propconfmat/100)
      
# ==========================================================================

import calculateWeightUsingGa2 as aresult
# weightvalga=aresult.getbestvalues(acc) 
weightvalga=aresult.getbestvalues(hd_clf) 

finalval=0
for i in range(len(acc)):
    finalval += weightvalga[i]*ypredproba_all[i]

print('f1_score',m.f1_score(ytest,np.argmax(finalval,axis=1),average='weighted'))
print('accuracy_score',m.accuracy_score(ytest,np.argmax(finalval,axis=1)))
          
# ==========================================================================

accuracies = pd.DataFrame(np.column_stack((acc,hd_clf)),columns=['acc','hd_clf'])
accuracies['val'] = 0.5*(accuracies.iloc[:,0] + accuracies.iloc[:,1])

finalval=0
for i in range(len(acc)):
    finalval += accuracies.val[i]*ypredproba_all[i]

print('f1_score',m.f1_score(ytest,np.argmax(finalval,axis=1),average='weighted'))
print('accuracy_score',m.accuracy_score(ytest,np.argmax(finalval,axis=1))) 

# ==========================================================================


ypredproba_all


























 