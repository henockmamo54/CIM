# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 01:02:52 2020

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

data=datasets.load_iris()

x= data.data
y=data.target
y[y!=2]=5
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
svc=SVC(random_state=randomseed,gamma='scale', probability=True,C=1)
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


temp=np.zeros((ytest.shape[0],2))
temp[:,0]=(ypredproba_all[0][:,0]*ypredconfprob_all[0][0][0] + ypredproba_all[0][:,0]*ypredconfprob_all[0][0][1]+     
    ypredproba_all[1][:,0]*ypredconfprob_all[1][0][0] + ypredproba_all[1][:,0]*ypredconfprob_all[1][0][1]+       
    ypredproba_all[2][:,0]*ypredconfprob_all[2][0][0] + ypredproba_all[2][:,0]*ypredconfprob_all[2][0][1]
    )

temp[:,1]=(ypredproba_all[0][:,1]*ypredconfprob_all[0][1][0] + ypredproba_all[0][:,1]*ypredconfprob_all[0][1][1]+        
    ypredproba_all[1][:,1]*ypredconfprob_all[1][1][0] + ypredproba_all[1][:,1]*ypredconfprob_all[1][1][1]+        
    ypredproba_all[2][:,1]*ypredconfprob_all[2][1][0] + ypredproba_all[2][:,1]*ypredconfprob_all[2][1][1]
    )

temp=np.argmax(temp,axis=1)    

temp[temp==1]=5
temp[temp==0]=2

print(m.accuracy_score(ytest,temp))




#
#
##===============================
#
#
#
##=============================== class 2 modles
#ytrain1=ytrain.copy()
#ytest1=ytest.copy()
#ytrain1[ytrain1!=1]=5
#ytest1[ytest1!=1]=5
#
#rf1=RandomForestClassifier(random_state=randomseed,n_estimators=100)
#rf1.fit(xtrain,ytrain1)
#ypred1=rf1.predict(xtest)
#ytrain_1_pred=rf1.predict_proba(xtrain)
#print('Class 1 Accuracy',m.accuracy_score(ytest1,ypred1))
#
#ypredprob1 = rf1.predict_proba(xtest)
#conf1=m.confusion_matrix(ytest1,ypred1)
#conf1_h=np.sum(conf1,axis=1)
#
#for i in range(len(conf1_h)):
#    conf1[i]=100*conf1[i]/conf1_h[i]
#conf1=conf1/100
#
##===============================
#
#
#
##=============================== class 1 modles
#ytrain2=ytrain.copy()
#ytest2=ytest.copy()
#
#ytrain2[ytrain2!=2]=5
#ytest2[ytest2!=2]=5
#
#rf2=RandomForestClassifier(random_state=randomseed,n_estimators=100)
#rf2.fit(xtrain,ytrain2)
#ypred2=rf2.predict(xtest)
#ytrain_2_pred=rf2.predict_proba(xtrain)
#print('Class 2 Accuracy',m.accuracy_score(ytest2,ypred2))
#
#ypredprob2 = rf2.predict_proba(xtest)
#conf2=m.confusion_matrix(ytest2,ypred2)
#conf2_h=np.sum(conf2,axis=1)
#
#for i in range(len(conf2_h)):
#    conf2[i]=100*conf2[i]/conf2_h[i]
#conf2=conf2/100
#
##===============================
#
#
##===================*****************=============
##compute new probability for reach test value 
##prepare the data set for training
#
#import pandas as pd
#
##traindata
#traindata=pd.DataFrame(ytrain_0_pred);
#traindata['PR']=conf0[0][0]
##traindata['NR']=conf0[0][1]
#
#traindata['c1_0']=ytrain_1_pred[:,0]
#traindata['c1_1']=ytrain_1_pred[:,1]
#traindata['PR_1']=conf1[0][0]
##traindata['NR_1']=conf1[0][1]
#
#traindata['c2_0']=ytrain_2_pred[:,0]
#traindata['c2_1']=ytrain_2_pred[:,1]
#traindata['PR_2']=conf2[0][0]
##traindata['NR_2']=conf2[0][1]
#
##====testdata
#testdata=pd.DataFrame(ypredprob0);
#testdata['PR']=conf0[0][0]
##testdata['NR']=conf0[0][1]
#
#testdata['c1_0']=ypredprob1[:,0]
#testdata['c1_1']=ypredprob1[:,1]
#testdata['PR_1']=conf1[0][0]
##testdata['NR_1']=conf1[0][1]
#
#testdata['c2_0']=ypredprob2[:,0]
#testdata['c2_1']=ypredprob2[:,1]
#testdata['PR_2']=conf2[0][0]
##testdata['NR_2']=conf2[0][1]
#
#
#
#metaclf=RandomForestClassifier(random_state=randomseed,n_estimators=1000)
#metaclf.fit(traindata,ytrain)
#
#print('Final Accuracy',m.accuracy_score(ytest,metaclf.predict(testdata)))
#
##====================================================
#
#
#
#temp=np.zeros((ytest.shape[0],3))
#temp[:,0]=ypredprob0[:,0]*conf0[0][0] + ypredprob0[:,0]*conf0[0][1]
#temp[:,1]=ypredprob1[:,0]*conf1[0][0] + ypredprob1[:,0]*conf1[0][1]
#temp[:,2]=ypredprob2[:,0]*conf2[0][0] + ypredprob2[:,0]*conf2[0][1]
#
#print(m.accuracy_score(ytest,np.argmax(temp,axis=1)))
#
#
#
#temp2=np.zeros((ytest.shape[0],3))
#temp2[:,0]=ypredprob0[:,0]
#temp2[:,1]=ypredprob1[:,0]
#temp2[:,2]=ypredprob2[:,0]
#
#print(m.accuracy_score(ytest,np.argmax(temp2,axis=1)))
#
#















