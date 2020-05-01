# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:56:54 2020
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
import xgboost as xgb 
import warnings
warnings.filterwarnings('ignore')

randomseed=42
np.random.seed(randomseed)

clfs=[]

# ***************************************************************************
xtest= (pd.read_csv('xtest.txt'))
xtrain= (pd.read_csv('xtrain.txt'))
ytest= (pd.read_csv('ytest.txt')) 
ytrain= (pd.read_csv('ytrain.txt')) 

classval=0

xtrain=xtrain.drop(ytrain[ytrain.iloc[:,0]==classval].index)
ytrain=ytrain.drop(ytrain[ytrain.iloc[:,0]==classval].index) 
xtest=xtest.drop(ytest[ytest.iloc[:,0]==classval].index)
ytest=ytest.drop(ytest[ytest.iloc[:,0]==classval].index)  

ytrain_original=ytrain.copy()
ytest_original = ytest.copy() 

x_temp=np.concatenate((xtrain,xtest))
y_temp=np.concatenate((ytrain,ytest)).ravel()

temp=pd.DataFrame(instance_hardness.kdn_score(x_temp, y_temp, 12)[0],columns=['val'])


# #================================================= 


xtrain=np.column_stack((xtrain,temp.iloc[:-400,:]))
xtest=np.column_stack((xtest,temp.iloc[-400:,:]))
ytrain=np.column_stack((ytrain,temp.iloc[:-400,:]))
ytest=np.column_stack((ytest,temp.iloc[-400:,:]))


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain[:,:-1],ytrain[:,:-1])

xgbpred=xgbc.predict(xtest[:,:-1])
clfs.append(xgbc)
print('xgbpred ',m.f1_score(ytest[:,:-1],xgbpred,average='weighted'))

# #================================================= 

result=pd.DataFrame(np.column_stack((ytest,xgbpred)),columns=['actual','measure','xgb'])

hardp=result[result.measure>0.39]
softp=result[result.measure<0.39]

print('xgb hard score ',m.accuracy_score(hardp.actual,hardp.xgb))
print('xgb soft score ',m.accuracy_score(softp.actual,softp.xgb))


# ***************************************************************************


# ***************************************************************************
xtest= (pd.read_csv('xtest.txt'))
xtrain= (pd.read_csv('xtrain.txt'))
ytest= (pd.read_csv('ytest.txt')) 
ytrain= (pd.read_csv('ytrain.txt')) 

classval=1

xtrain=xtrain.drop(ytrain[ytrain.iloc[:,0]==classval].index)
ytrain=ytrain.drop(ytrain[ytrain.iloc[:,0]==classval].index) 
xtest=xtest.drop(ytest[ytest.iloc[:,0]==classval].index)
ytest=ytest.drop(ytest[ytest.iloc[:,0]==classval].index)  

ytrain_original=ytrain.copy()
ytest_original = ytest.copy() 

x_temp=np.concatenate((xtrain,xtest))
y_temp=np.concatenate((ytrain,ytest)).ravel()

temp=pd.DataFrame(instance_hardness.kdn_score(x_temp, y_temp, 12)[0],columns=['val'])


# #================================================= 


xtrain=np.column_stack((xtrain,temp.iloc[:-400,:]))
xtest=np.column_stack((xtest,temp.iloc[-400:,:]))
ytrain=np.column_stack((ytrain,temp.iloc[:-400,:]))
ytest=np.column_stack((ytest,temp.iloc[-400:,:]))


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain[:,:-1],ytrain[:,:-1])

xgbpred=xgbc.predict(xtest[:,:-1])
clfs.append(xgbc)
print('xgbpred ',m.f1_score(ytest[:,:-1],xgbpred,average='weighted'))

# #================================================= 

result=pd.DataFrame(np.column_stack((ytest,xgbpred)),columns=['actual','measure','xgb'])

hardp=result[result.measure>0.39]
softp=result[result.measure<0.39]

print('xgb hard score ',m.accuracy_score(hardp.actual,hardp.xgb))
print('xgb soft score ',m.accuracy_score(softp.actual,softp.xgb))


# ***************************************************************************

# ***************************************************************************
xtest= (pd.read_csv('xtest.txt'))
xtrain= (pd.read_csv('xtrain.txt'))
ytest= (pd.read_csv('ytest.txt')) 
ytrain= (pd.read_csv('ytrain.txt')) 

classval=2

xtrain=xtrain.drop(ytrain[ytrain.iloc[:,0]==classval].index)
ytrain=ytrain.drop(ytrain[ytrain.iloc[:,0]==classval].index) 
xtest=xtest.drop(ytest[ytest.iloc[:,0]==classval].index)
ytest=ytest.drop(ytest[ytest.iloc[:,0]==classval].index)  

ytrain_original=ytrain.copy()
ytest_original = ytest.copy() 

x_temp=np.concatenate((xtrain,xtest))
y_temp=np.concatenate((ytrain,ytest)).ravel()

temp=pd.DataFrame(instance_hardness.kdn_score(x_temp, y_temp, 12)[0],columns=['val'])


# #================================================= 


xtrain=np.column_stack((xtrain,temp.iloc[:-400,:]))
xtest=np.column_stack((xtest,temp.iloc[-400:,:]))
ytrain=np.column_stack((ytrain,temp.iloc[:-400,:]))
ytest=np.column_stack((ytest,temp.iloc[-400:,:]))


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain[:,:-1],ytrain[:,:-1])

xgbpred=xgbc.predict(xtest[:,:-1])
clfs.append(xgbc)
print('xgbpred ',m.f1_score(ytest[:,:-1],xgbpred,average='weighted'))

# #================================================= 

result=pd.DataFrame(np.column_stack((ytest,xgbpred)),columns=['actual','measure','xgb'])

hardp=result[result.measure>0.39]
softp=result[result.measure<0.39]

print('xgb hard score ',m.accuracy_score(hardp.actual,hardp.xgb))
print('xgb soft score ',m.accuracy_score(softp.actual,softp.xgb))


# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************


xtest= (pd.read_csv('xtest.txt'))
xtrain= (pd.read_csv('xtrain.txt'))
ytest= (pd.read_csv('ytest.txt')) 
ytrain= (pd.read_csv('ytrain.txt'))   

ytrain_original=ytrain.copy()
ytest_original = ytest.copy() 

x_temp=np.concatenate((xtrain,xtest))
y_temp=np.concatenate((ytrain,ytest)).ravel()

temp=pd.DataFrame(instance_hardness.kdn_score(x_temp, y_temp, 12)[0],columns=['val'])


# #================================================= 


xtrain=np.column_stack((xtrain,temp.iloc[:-600,:]))
xtest=np.column_stack((xtest,temp.iloc[-600:,:]))
ytrain=np.column_stack((ytrain,temp.iloc[:-600,:]))
ytest=np.column_stack((ytest,temp.iloc[-600:,:]))

ytest=pd.DataFrame(ytest)
index_= ytest[ytest.iloc[:,1]>0.29].index

xtest=np.array(pd.DataFrame(xtest).drop(index_))
ytest=np.array(ytest.drop(index_))


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain[:,:-1],ytrain[:,:-1])

xgbpred=xgbc.predict(xtest[:,:-1])
clfs.append(xgbc)
print('xgbpred ',m.f1_score(ytest[:,:-1],xgbpred,average='weighted'))

# #================================================= 

pred0=clfs[0].predict_proba(xtest[:,:-1]) 
pred1=clfs[1].predict_proba(xtest[:,:-1]) 
pred2=clfs[2].predict_proba(xtest[:,:-1]) 

pred0_=np.argmax(pred0,axis=1)
pred0_[pred0_==1]=2
pred0_[pred0_==0]=1

pred1_=np.argmax(pred1,axis=1)
pred1_[pred1_==1]=2

pred2_=np.argmax(pred2,axis=1)

pred=pd.DataFrame(np.column_stack((pred0_,pred1_,pred2_)))
pred00_=pred.mode(axis=1)


print(m.accuracy_score(ytest[:,0],pred00_))

# #================================================= 

pred_c0 = pred1[:,0] + pred2[:,0] 
pred_c1 = pred0[:,0] + pred2[:,1] 
pred_c2 = pred0[:,1] + pred1[:,1] 

predc_=np.argmax(np.column_stack((pred_c0,pred_c1,pred_c2)),axis=1)
print(m.accuracy_score(ytest[:,0],predc_))

# #================================================= 

pred_c0 = 0.70*pred1[:,0] + 0.54*pred2[:,0] 
pred_c1 = 0.70*pred0[:,0] + 0.54*pred2[:,1] 
pred_c2 = 0.70*pred0[:,1] + 0.70*pred1[:,1] 

predc_=np.argmax(np.column_stack((pred_c0,pred_c1,pred_c2)),axis=1)
print(m.accuracy_score(ytest[:,0],predc_))

# #================================================= 


















