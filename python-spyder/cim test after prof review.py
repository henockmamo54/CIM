# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:40:34 2020

@author: Henock
"""



from sklearn import datasets
from sklearn import metrics as m
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score 
import numpy as np
from itertools import combinations
from sklearn import datasets
from sklearn import metrics as m
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score 
import numpy as np
import pandas as pd 
from itertools import combinations
from sklearn.svm import SVC
import xgboost as xgb
from thundersvm import SVC as svmgpu


randomseed=10
np.random.seed(randomseed)


xtest=np.array(pd.read_csv('xtest.txt'))
xtrain=np.array(pd.read_csv('xtrain.txt'))
ytest_original=np.array(pd.read_csv('ytest.txt')).ravel()
ytrain_original =np.array(pd.read_csv('ytrain.txt')).ravel()

clf=[]
acc=[]
finalacc=[]
ypredproba_all=[]
ypredconfprob_all=[]

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()

rf=RandomForestClassifier(random_state=randomseed, n_estimators=100)
rf.fit(xtrain,ytrain)
print('original score',m.accuracy_score(ytest,rf.predict(xtest)))



#=================================================
# classs 0
#=================================================

def swapcolumns(trainval,testval,coldindexval):
    trainval[trainval!=coldindexval]=5
    testval[testval!=coldindexval]=5
    
    trainval[trainval==coldindexval]=0
    trainval[trainval==5]=1
    
    testval[testval==coldindexval]=0
    testval[testval==5]=1
    
    return trainval,testval

#=================================================
# classs 0
#=================================================
    
ytrain,ytest= swapcolumns(ytrain,ytest,2)
#=================================================

rf=RandomForestClassifier(random_state=randomseed, n_estimators=100)
rf.fit(xtrain,ytrain)
rfpred=rf.predict(xtest)
print(m.accuracy_score(ytest,rfpred))

clf.append(rf)
acc.append(m.accuracy_score(ytest,rfpred))
ypredproba_all.append(rf.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,rfpred)
confsumh=np.sum(confmat,axis=0)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[:,i]= 100*propconfmat[:,i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)

#=================================================
svc=svmgpu(random_state=randomseed,probability=True,C=100,gamma=1)
svc.fit(xtrain,ytrain)

svcpred=svc.predict(xtest)
print(m.accuracy_score(ytest,svcpred))

clf.append(svc)
acc.append(m.accuracy_score(ytest,svcpred))
ypredproba_all.append(1-svc.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,svcpred)
confsumh=np.sum(confmat,axis=0)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[:,i]= 100*propconfmat[:,i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)

#=================================================
xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=10)
xgbc.fit(xtrain,ytrain)

xgbpred=xgbc.predict(xtest)
print(m.accuracy_score(ytest,xgbpred))


clf.append(xgbc)
acc.append(m.accuracy_score(ytest,xgbpred))
ypredproba_all.append(xgbc.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,xgbpred)
confsumh=np.sum(confmat,axis=0)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[:,i]= 100*propconfmat[:,i]/confsumh[i]  
ypredconfprob_all.append(propconfmat/100)


acc=acc/np.sum(acc)
temp=np.zeros((ytest.shape[0],2))
temp[:,0]=(
	acc[0]*(ypredproba_all[0][:,0]*ypredconfprob_all[0][0][0]	+	ypredproba_all[0][:,1]*ypredconfprob_all[0][0][1])	+  
	acc[1]*(ypredproba_all[1][:,0]*ypredconfprob_all[1][0][0]	+	ypredproba_all[1][:,1]*ypredconfprob_all[1][0][1])	+ 
	acc[2]*(ypredproba_all[2][:,0]*ypredconfprob_all[2][0][0]	+	ypredproba_all[2][:,1]*ypredconfprob_all[2][0][1]) 
)

temp[:,1]=(
	acc[0]*(ypredproba_all[0][:,0]*ypredconfprob_all[0][1][0]	+	ypredproba_all[0][:,1]*ypredconfprob_all[0][1][1])	+        
	acc[1]*(ypredproba_all[1][:,0]*ypredconfprob_all[1][1][0]	+	ypredproba_all[1][:,1]*ypredconfprob_all[1][1][1])	+        
	acc[2]*(ypredproba_all[2][:,0]*ypredconfprob_all[2][1][0]	+	ypredproba_all[2][:,1]*ypredconfprob_all[2][1][1])
    )
	
temp0=temp.copy()/3
temp=np.argmax(temp,axis=1)    


temp0conf=m.confusion_matrix(ytest,temp)
temp0confh=np.sum(temp0conf,axis=1)

for i in range(temp0conf.shape[0]):
    temp0conf[i]= 100*temp0conf[i]/temp0confh[i] 
temp0conf=temp0conf/100

print('classs 0 ',m.accuracy_score(ytest,temp))
finalacc.append(m.accuracy_score(ytest,temp))
