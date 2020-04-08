# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:59:52 2020

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
from sklearn.utils import shuffle
import itertools
from sklearn.ensemble import VotingClassifier




randomseed=10
np.random.seed(randomseed) 

xtest=np.array(pd.read_csv('xtest.txt'))
xtrain=np.array(pd.read_csv('xtrain.txt'))
ytest_original=np.array(pd.read_csv('ytest.txt')).ravel()
ytrain_original =np.array(pd.read_csv('ytrain.txt')).ravel()

#data=datasets.load_iris()
#x=data.data
#y=data.target
#xtrain,xtest,ytrain_original,ytest_original=train_test_split(x,y,random_state=randomseed,test_size=0.3) 

rf=RandomForestClassifier(random_state=randomseed, n_estimators=100)
rf.fit(xtrain,ytrain_original)
print('original score',m.accuracy_score(ytest_original,rf.predict(xtest)))


clf=[]
acc=[]
ypredproba_all=[]
ypredconfprob_all=[]

acc_train=[]
ypredproba_all_train=[]
ypredconfprob_all_train=[]

ytrain=ytrain_original.copy()
ytest=ytest_original.copy() 

clf = []
acc = []
ypredproba_all = []
ypredconfprob_all = []
acc_train = []


# classifier 1
# ===========================

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


# classifier 2
# ===========================
svc = svmgpu(random_state=randomseed, probability=True, C=1)
svc.fit(xtrain, ytrain)

svcpred = svc.predict(xtest)
print(m.accuracy_score(ytest, svcpred))

clf.append(svc)
acc.append(m.accuracy_score(ytest, svcpred))
ypredproba_all.append(1 - svc.predict_proba(xtest))

confmat = m.confusion_matrix(ytest, svcpred)
confsumh = np.sum(confmat, axis=0)
propconfmat = confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[:, i] = 100 * propconfmat[:, i] / confsumh[i]
ypredconfprob_all.append(propconfmat / 100)

# classifier 3
# ===========================
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

ci0=ci1=ci2=1

p_k1_c0=((ypredproba_all[0][:,0] * ypredconfprob_all[0][0][0]   + 
              ypredproba_all[0][:,1] * ypredconfprob_all[0][0][1] ) +
              ypredproba_all[0][:,2] * ypredconfprob_all[0][0][2] ) 

p_k1_c1=((ypredproba_all[0][:,0] * ypredconfprob_all[0][1][0]   + 
              ypredproba_all[0][:,1] * ypredconfprob_all[0][1][1] ) +
              ypredproba_all[0][:,2] * ypredconfprob_all[0][1][2] )   

p_k1_c2=((ypredproba_all[0][:,0] * ypredconfprob_all[0][2][0]   + 
              ypredproba_all[0][:,1] * ypredconfprob_all[0][2][1] ) +
              ypredproba_all[0][:,2] * ypredconfprob_all[0][2][2] )  
 


p_k2_c0=((ypredproba_all[1][:,0] * ypredconfprob_all[1][0][0]   + 
              ypredproba_all[1][:,1] * ypredconfprob_all[1][0][1] ) +
              ypredproba_all[1][:,2] * ypredconfprob_all[1][0][2] ) 

p_k2_c1=((ypredproba_all[1][:,0] * ypredconfprob_all[1][1][0]   + 
              ypredproba_all[1][:,1] * ypredconfprob_all[1][1][1] ) +
              ypredproba_all[1][:,2] * ypredconfprob_all[1][1][2] )   

p_k2_c2=((ypredproba_all[1][:,0] * ypredconfprob_all[1][2][0]   + 
              ypredproba_all[1][:,1] * ypredconfprob_all[1][2][1] ) +
              ypredproba_all[1][:,2] * ypredconfprob_all[1][2][2] )  
 
    

p_k3_c0=((ypredproba_all[2][:,0] * ypredconfprob_all[2][0][0] + 
              ypredproba_all[2][:,1] * ypredconfprob_all[2][0][1] ) +
              ypredproba_all[2][:,2] * ypredconfprob_all[2][0][2] ) 

p_k3_c1=((ypredproba_all[2][:,0] * ypredconfprob_all[2][1][0]   + 
              ypredproba_all[2][:,1] * ypredconfprob_all[2][1][1] ) +
              ypredproba_all[2][:,2] * ypredconfprob_all[2][1][2] )   

p_k3_c2=((ypredproba_all[2][:,0] * ypredconfprob_all[2][2][0]   + 
              ypredproba_all[2][:,1] * ypredconfprob_all[2][2][1] ) +
              ypredproba_all[2][:,2] * ypredconfprob_all[2][2][2] )  
 
 
pc1 = p_k1_c0 + p_k2_c0 + p_k3_c0
pc2 = p_k1_c1 + p_k2_c1 + p_k3_c1
pc3 = p_k1_c2 + p_k2_c2 + p_k3_c2


finalcol = np.zeros((ytest.shape[0], 3))
finalcol[:, 0] = pc1
finalcol[:, 1] = pc2
finalcol[:, 2] = pc3
finalpred = np.argmax(finalcol, axis=1)

ytest = ytest_original.copy()
print(m.accuracy_score(ytest, finalpred))
print(m.confusion_matrix(ytest, finalpred))
