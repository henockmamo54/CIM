# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:43:49 2020

@author: Henock
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:10:03 2020

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
from itertools import combinations
from sklearn.svm import SVC
import xgboost as xgb


randomseed=42
np.random.seed(randomseed)

iris=datasets.load_iris()

x= iris.data
y=iris.target
xtest,xtrain,ytest,ytrain=train_test_split(x,y,random_state=randomseed,train_size=0.2)

clf=[]
acc=[]
ypredproba_all=[]
ypredconfprob_all=[]

#=================================================
rf=RandomForestClassifier(random_state=randomseed, n_estimators=100)
rf.fit(xtrain,ytrain)
rfpred=rf.predict(xtest)
print(m.accuracy_score(ytest,rfpred))

clf.append(rf)
acc.append(m.accuracy_score(ytest,rfpred))
ypredproba_all.append(rf.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,rfpred)
confsumh=np.sum(confmat,axis=1)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)

#=================================================
svc=SVC(random_state=randomseed,gamma='scale', probability=True)
svc.fit(xtrain,ytrain)

svcpred=svc.predict(xtest)
print(m.accuracy_score(ytest,svcpred))


clf.append(svc)
acc.append(m.accuracy_score(ytest,svcpred))
ypredproba_all.append(svc.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,svcpred)
confsumh=np.sum(confmat,axis=1)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
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
confsumh=np.sum(confmat,axis=1)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)

#=================================================


#probability being class 0
pbc0=[]
c=0
for i in range(xtest.shape[0]):
    val=0
    for j in range(len(clf)): # classifiers
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][0]
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][1]
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][2]
    pbc0.append(val)
        

#probability being class 1
pbc1=[]
c=1
for i in range(xtest.shape[0]):
    val=0
    for j in range(len(clf)): # classifiers
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][0]
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][1]
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][2]
    pbc1.append(val)
        
#probability being class 2
pbc2=[]
c=2
for i in range(xtest.shape[0]):
    val=0
    for j in range(len(clf)): # classifiers
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][0]
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][1]
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][2]
    pbc2.append(val)
        

temp=np.zeros((len(pbc0),3))
temp[:,0]=pbc0
temp[:,1]=pbc1
temp[:,2]=pbc2
        
print(m.accuracy_score(ytest,np.argmax(temp,axis=1)))
    



#=================================================



