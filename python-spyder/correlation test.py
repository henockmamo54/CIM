# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:28:33 2020

@author: Henock
"""

import  numpy as np
from sklearn import datasets
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

def swapcolumns(trainval, testval, coldindexval):
    trainval[trainval != coldindexval] = 5
    testval[testval != coldindexval] = 5

    trainval[trainval == coldindexval] = 0
    trainval[trainval == 5] = 1

    testval[testval == coldindexval] = 0
    testval[testval == 5] = 1

    return trainval, testval


randomseed=42
np.random.seed(randomseed)

xtest=np.array(pd.read_csv('xtest.txt'))
xtrain=np.array(pd.read_csv('xtrain.txt'))
ytest_original=np.array(pd.read_csv('ytest.txt')).ravel()
ytrain_original =np.array(pd.read_csv('ytrain.txt')).ravel()

# data=datasets.load_wine()
# x=data.data
# y=data.target

# data = pd.read_csv("../dataset/seeds_dataset.txt", sep="\t", header=None)
# data = shuffle(data)
# le = LabelEncoder()
# data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
# x = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# print(np.unique(y))

# data = pd.read_csv("../dataset/ionosphere.data",  header=None)
# data = shuffle(data)
# le = LabelEncoder()
# data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
# x = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# print(np.unique(y))




# xtrain,xtest,ytrain_original,ytest_original=train_test_split(x,y,random_state=randomseed,test_size=0.3) 

ytrain=ytrain_original.copy()
ytest=ytest_original.copy() 

# ytrain, ytest = swapcolumns(ytrain, ytest, 2)



clf=[]
acc=[]
finalacc=[]
ypredproba_all=[]
ypredconfprob_all=[]

rf=RandomForestClassifier(random_state=randomseed, n_estimators=10)
rf.fit(xtrain,ytrain)
print('original score',m.f1_score(ytest,rf.predict(xtest),average='weighted'))


xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain,ytrain)
xgbpred=xgbc.predict(xtest)
print(m.f1_score(ytest,xgbpred,average='weighted'))


#================================================= 

# Class 0
# ===========================
ytrain = ytrain_original.copy()
ytest = ytest_original.copy()
ytrain,ytest= swapcolumns(ytrain,ytest,0)
#=================================================
 
xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain,ytrain)

xgbpred=xgbc.predict(xtest)
print(m.f1_score(ytest,xgbpred,average='weighted'))


clf.append(xgbc)
acc.append(m.f1_score(ytest,xgbpred,average='weighted'))
ypredproba_all.append(xgbc.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,xgbpred)
confsumh=np.sum(confmat,axis=1)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)

# =================================================
# classs 1
# =================================================
ytrain = ytrain_original.copy()
ytest = ytest_original.copy()
ytrain, ytest = swapcolumns(ytrain, ytest, 1)
# =================================================
  
xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain,ytrain)

xgbpred=xgbc.predict(xtest)
print(m.f1_score(ytest,xgbpred,average='weighted'))


clf.append(xgbc)
acc.append(m.f1_score(ytest,xgbpred,average='weighted'))
ypredproba_all.append(xgbc.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,xgbpred)
confsumh=np.sum(confmat,axis=1)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)

#=================================================
# classs 2
#=================================================

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()
ytrain,ytest= swapcolumns(ytrain,ytest,2)

#=================================================
 
xgbc=xgb.XGBClassifier(random_state=randomseed,n_estimators=100)
xgbc.fit(xtrain,ytrain)

xgbpred=xgbc.predict(xtest)
print(m.f1_score(ytest,xgbpred,average='weighted'))


clf.append(xgbc)
acc.append(m.f1_score(ytest,xgbpred,average='weighted'))
ypredproba_all.append(xgbc.predict_proba(xtest))

confmat=m.confusion_matrix(ytest,xgbpred)
confsumh=np.sum(confmat,axis=1)
propconfmat=confmat.copy()
for i in range(propconfmat.shape[0]):
    propconfmat[i]= 100*propconfmat[i]/confsumh[i] 
ypredconfprob_all.append(propconfmat/100)


#=================================================
#=================================================
#=================================================

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()
ytrain,ytest_0= swapcolumns(ytrain,ytest,0)

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()
ytrain,ytest_1= swapcolumns(ytrain,ytest,1)

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()
ytrain,ytest_2= swapcolumns(ytrain,ytest,2)


propdata=pd.DataFrame()

propdata['ytest_0'] =ytest_0
propdata['ytest_1'] =ytest_1
propdata['ytest_2'] =ytest_2

propdata['C1_p'] = np.array(ypredproba_all)[0,:,0]
propdata['C1_N'] = np.array(ypredproba_all)[0,:,1]
propdata['C2_p'] = np.array(ypredproba_all)[1,:,0]
propdata['C2_N'] = np.array(ypredproba_all)[1,:,1]
propdata['C3_p'] = np.array(ypredproba_all)[2,:,0]
propdata['C3_N'] = np.array(ypredproba_all)[2,:,1]
propdata['C1_N_C2'] = np.array(ypredproba_all)[0,:,1]*np.array(ypredproba_all)[1,:,1]
propdata['C1_N_C3'] = np.array(ypredproba_all)[0,:,1]*np.array(ypredproba_all)[2,:,1]
propdata['C2_N_C3'] = np.array(ypredproba_all)[1,:,1]*np.array(ypredproba_all)[2,:,1]


# propdata['class1'] = propdata.C2_N_C3 * propdata.C1_p
# propdata['class2'] = propdata.C1_N_C3 * propdata.C2_p
# propdata['class3'] = propdata.C1_N_C2 * propdata.C3_p

# propdata['class1'] = propdata.C1_N 
# propdata['class2'] = propdata.C2_N
# propdata['class3'] = propdata.C3_N 


# propdata['class1'] = 0.976*propdata.C2_N_C3 + propdata.C1_p
# propdata['class2'] = 0.982*propdata.C1_N_C3 + propdata.C2_p
# propdata['class3'] = 0.974*propdata.C1_N_C2 + propdata.C3_p


ytrain=ytrain_original.copy()
ytest=ytest_original.copy()
finalcol = np.zeros((ytest.shape[0], 3))
finalcol[:, 0] = propdata.class1
finalcol[:, 1] = propdata.class2
finalcol[:, 2] = propdata.class3
temp=np.argmax(finalcol,axis=1) 
print(m.accuracy_score(ytest,temp))



corrdata = abs(propdata.corr())
import seaborn as sns
from matplotlib import pyplot as plt
sns.heatmap(corrdata, annot=True, cmap=plt.cm.Reds)
plt.show()


# #=================================================



ytrain=ytrain_original.copy()
ytest=ytest_original.copy()
ytrain,ytest= swapcolumns(ytrain,ytest,1)

# temp=np.argmax(ypredproba_all[1],axis=1)
temp = np.argmax(np.column_stack((propdata.C2_p,propdata.C2_N)),axis=1)
# temp = np.argmax(np.column_stack((propdata.C2_p,propdata.C2_N)),axis=1)

# temp = np.argmax(np.column_stack((np.max(np.column_stack((0.91*propdata.C2_p,
#                                                           1*0.95*propdata.C1_N_C3)),axis=1)
#                                   ,propdata.C2_N)),axis=1)

print(m.accuracy_score(ytest,temp))

# #=================================================

import calculateWeightUsingGa2 as aresult
weightvalga=aresult.getbestvalues(acc)

finalcol = np.zeros((ytest.shape[0], 3))
finalcol[:, 0] = ypredproba_all[0][:, 0]*weightvalga[0]
finalcol[:, 1] = ypredproba_all[1][:, 0]*weightvalga[1]
finalcol[:, 2] = ypredproba_all[2][:, 0]*weightvalga[2]

finalpred = np.argmax(finalcol, axis=1)

ytest = ytest_original.copy()
print('accuracy_score',m.accuracy_score(ytest, finalpred))
# accuracy_score 0.6433333333333333

# # #=================================================
# #=================================================

import calculateWeightUsingGa2 as aresult
weightvalga=aresult.getbestvalues(acc)

finalcol = np.zeros((ytest.shape[0], 3))
finalcol[:, 0] = ypredproba_all[0][:, 0]*weightvalga[0]*propdata.C2_N_C3
finalcol[:, 1] = ypredproba_all[1][:, 0]*weightvalga[1]*propdata.C1_N_C3
finalcol[:, 2] = ypredproba_all[2][:, 0]*weightvalga[2]*propdata.C1_N_C2

finalpred = np.argmax(finalcol, axis=1)

ytest = ytest_original.copy()
print('accuracy_score',m.accuracy_score(ytest, finalpred))
# accuracy_score 0.6433333333333333

# # #=================================================

# import calculateWeightUsingGa2 as aresult
# weightvalga=aresult.getbestvalues([
#     ypredconfprob_all[0][0][0] * ypredconfprob_all[1][1][1]*ypredconfprob_all[2][1][1],
#     ypredconfprob_all[1][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[2][1][1],
#     ypredconfprob_all[2][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[1][1][1]])

# finalcol = np.zeros((ytest.shape[0], 3))
# finalcol[:, 0] = ypredproba_all[0][:, 0]*weightvalga[0]
# finalcol[:, 1] = ypredproba_all[1][:, 0]*weightvalga[1]
# finalcol[:, 2] = ypredproba_all[2][:, 0]*weightvalga[2]

# finalpred = np.argmax(finalcol, axis=1)

# ytest = ytest_original.copy()
# print('accuracy_score',m.accuracy_score(ytest, finalpred))
# # accuracy_score 0.6783333333333333
# # accuracy_score 0.715
# # accuracy_score 0.7216666666666667

# # #=================================================

# import calculateWeightUsingGa2 as aresult
# weightvalga=aresult.getbestvalues([
#     ypredconfprob_all[0][0][0] * ypredconfprob_all[1][1][1]*ypredconfprob_all[2][1][1],
#     ypredconfprob_all[1][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[2][1][1],
#     ypredconfprob_all[2][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[1][1][1]])

# finalcol = np.zeros((ytest.shape[0], 3))
# finalcol[:, 0] = ypredproba_all[0][:, 0] * weightvalga[0] + ypredproba_all[0][:, 1] * ypredconfprob_all[0][0][1]
# finalcol[:, 1] = ypredproba_all[1][:, 0] * weightvalga[1] + ypredproba_all[1][:, 1] * ypredconfprob_all[1][0][1]
# finalcol[:, 2] = ypredproba_all[2][:, 0] * weightvalga[2] + ypredproba_all[2][:, 1] * ypredconfprob_all[2][0][1]

# finalpred = np.argmax(finalcol, axis=1)

# ytest = ytest_original.copy()
# print('accuracy_score',m.accuracy_score(ytest, finalpred))

# # #=================================================

# import calculateWeightUsingGa2 as aresult
# weightvalga=aresult.getbestvalues([
#     ypredconfprob_all[0][0][0] * ypredconfprob_all[1][1][1]*ypredconfprob_all[2][1][1] ,
#     -ypredconfprob_all[0][0][1],
    
#     ypredconfprob_all[1][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[2][1][1],
#     -ypredconfprob_all[1][0][1],
    
#     ypredconfprob_all[2][0][0] * ypredconfprob_all[0][1][1]*ypredconfprob_all[1][1][1],
#     -ypredconfprob_all[2][0][1],
    
#     ])

# finalcol = np.zeros((ytest.shape[0], 3))
# finalcol[:, 0] = ypredproba_all[0][:, 0]*weightvalga[0] + ypredproba_all[0][:, 1]*weightvalga[1]
# finalcol[:, 1] = ypredproba_all[1][:, 0]*weightvalga[2] + ypredproba_all[1][:, 1]*weightvalga[3]
# finalcol[:, 2] = ypredproba_all[2][:, 0]*weightvalga[4] + ypredproba_all[2][:, 1]*weightvalga[5]

# finalpred = np.argmax(finalcol, axis=1)

# ytest = ytest_original.copy()
# print('accuracy_score',m.accuracy_score(ytest, finalpred))


# # _rf_p=weightvalga[0]*ypredproba_all[0]
# # _svm_p=weightvalga[1]*ypredproba_all[1]
# # _xgb_p=weightvalga[2]*ypredproba_all[2]
# # _temp=np.argmax(_rf_p+_svm_p+_xgb_p,axis=1)
# # print(m.f1_score(ytest,_temp,average='weighted'))

# # finalval=0
# # for i in range(len(acc)):
# #     finalval += weightvalga[i]*ypredproba_all[i]

# # ytest = ytest_original.copy()

# # print('f1_score',m.f1_score(ytest,np.argmax(finalval,axis=1),average='weighted'))
# # print('accuracy_score',m.accuracy_score(ytest,np.argmax(finalval,axis=1)))
    








