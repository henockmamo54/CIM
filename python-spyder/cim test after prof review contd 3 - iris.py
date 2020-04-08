# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:01:37 2020

@author: Henock
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 02:17:13 2020

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
    
ytrain,ytest= swapcolumns(ytrain,ytest,0)

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
# classs 1
#=================================================
ytrain=ytrain_original.copy()
ytest=ytest_original.copy()
ytrain,ytest= swapcolumns(ytrain,ytest,1)
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
# classs 2
#=================================================

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()
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

#=================================================

#=================================================

finalcol=np.zeros((ytest.shape[0],3))

finalcol[:,0]=(ypredproba_all[0][:,0]*ypredconfprob_all[0][0][0] + (ypredproba_all[0][:,1]*ypredconfprob_all[0][0][1] ))
finalcol[:,1]=(ypredproba_all[1][:,0]*ypredconfprob_all[1][0][0] + (ypredproba_all[1][:,1]*ypredconfprob_all[1][0][1] ))
finalcol[:,2]=(ypredproba_all[2][:,0]*ypredconfprob_all[2][0][0] + (ypredproba_all[2][:,1]*ypredconfprob_all[2][0][1] ))

finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
#print( m.confusion_matrix(ytest,finalpred))

#=================================================

#=================================================

finalcol=np.zeros((ytest.shape[0],3))

finalcol[:,0]=(ypredproba_all[0][:,0]*ypredconfprob_all[0][0][0] )
finalcol[:,1]=(ypredproba_all[1][:,0]*ypredconfprob_all[1][0][0] )
finalcol[:,2]=(ypredproba_all[2][:,0]*ypredconfprob_all[2][0][0] )

finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
#print( m.confusion_matrix(ytest,finalpred))


#=================================================

#=================================================

#from matplotlib import pyplot as plt
#fig = plt.figure()
#
#plt.subplot(4, 2, 1)
#plt.hist(ypredproba_all[0][:,0],bins=10) 
#plt.subplot(4, 2, 2)
#plt.hist(ypredproba_all[0][:,1],bins=10) 
#
#plt.subplot(4, 2, 3)
#plt.hist(ypredproba_all[1][:,0],bins=10) 
#plt.subplot(4, 2, 4)
#plt.hist(ypredproba_all[1][:,1],bins=10)
#
#plt.subplot(4, 2, 5)
#plt.hist(ypredproba_all[2][:,0],bins=10) 
#plt.subplot(4, 2, 6)
#plt.hist(ypredproba_all[2][:,1],bins=10)
#
#plt.subplot(4, 2, 7)
#plt.hist(finalcol[:,0],bins=10) 
#plt.subplot(4, 2, 8)
#plt.hist(finalcol[:,1],bins=10)
#
#plt.show()



#=================================================
# prepare dataset
#=================================================

mydata=pd.DataFrame()
for i in range(3):
    mydata[str(i*6)] = ypredproba_all[i][:,0]
    mydata[str(i*6 +1)] = ypredproba_all[i][:,1]
    
    mydata[str(i*6 +2)] = ypredconfprob_all[i][0][0]
    mydata[str(i*6 +3)] = ypredconfprob_all[i][0][1]
    mydata[str(i*6 +4)] = ypredconfprob_all[i][1][0]
    mydata[str(i*6 +5)] = ypredconfprob_all[i][1][1]    
mydata['y']=ytest_original.copy()

from sklearn.utils import shuffle
mydata=shuffle(mydata)


#temp_xtrain,temp_xtest,temp_ytrain,temp_ytest=train_test_split(mydata.iloc[:,:-1],mydata.iloc[:,-1],random_state=randomseed)
#
#temp_rf=xgb.XGBClassifier(n_estimators=1000,random_state=randomseed)
#
#temp_rf.fit(temp_xtrain,temp_ytrain)
#print('accuracy',m.accuracy_score(temp_ytest,temp_rf.predict(temp_xtest)))






finalcol=np.zeros((ytest.shape[0],3))

#finalcol[:,0]=(ypredproba_all[0][:,0]*ypredconfprob_all[0][0][0] * ypredproba_all[1][:,1]*ypredconfprob_all[1][1][1] * ypredproba_all[2][:,1]*ypredconfprob_all[2][1][1] )
#finalcol[:,1]=(ypredproba_all[0][:,1] *ypredconfprob_all[0][1][1]  * ypredproba_all[1][:,0]*ypredconfprob_all[1][0][0] * ypredproba_all[2][:,1]*ypredconfprob_all[2][1][1] )
#finalcol[:,2]=(ypredproba_all[0][:,1] *ypredconfprob_all[0][1][1]  * ypredproba_all[1][:,1]*ypredconfprob_all[1][1][1] * ypredproba_all[2][:,0] *ypredconfprob_all[2][0][0])


#finalcol[:,0]=(	acc[0]*(ypredproba_all[0][:,0] * ypredconfprob_all[0][0][0] + ypredproba_all[0][:,1] * ypredconfprob_all[0][0][1] ) +
#				acc[1]*(ypredproba_all[1][:,1] * ypredconfprob_all[1][1][1] + ypredproba_all[1][:,0] * ypredconfprob_all[1][1][0] ) +
#                acc[2]*(ypredproba_all[2][:,1] * ypredconfprob_all[2][1][1] + ypredproba_all[2][:,0] * ypredconfprob_all[2][1][0] ) )
#				
#finalcol[:,1]=( acc[0]*(ypredproba_all[0][:,1] * ypredconfprob_all[0][1][1] + ypredproba_all[0][:,0] * ypredconfprob_all[0][1][0] ) +
#				acc[1]*(ypredproba_all[1][:,0] * ypredconfprob_all[1][0][0] + ypredproba_all[1][:,1] * ypredconfprob_all[1][1][0] ) +
#				acc[2]*(ypredproba_all[2][:,1] * ypredconfprob_all[2][1][1] + ypredproba_all[2][:,0] * ypredconfprob_all[2][1][0] ) )
#				
#finalcol[:,2]=( acc[0]*(ypredproba_all[0][:,1] * ypredconfprob_all[0][1][1] + ypredproba_all[0][:,0] * ypredconfprob_all[0][1][0] ) + 
#                acc[1]*(ypredproba_all[1][:,1] * ypredconfprob_all[1][1][1] + ypredproba_all[1][:,0] * ypredconfprob_all[1][1][0] ) +
#                acc[2]*(ypredproba_all[2][:,0] * ypredconfprob_all[2][0][0] + ypredproba_all[2][:,1] * ypredconfprob_all[2][0][1] ) )
#				
#
#

finalcol=np.zeros((ytest.shape[0],3))

ci0=np.sum(abs(ypredproba_all[0][:,0]-0.5))/ypredproba_all[0][:,0].shape[0]
ci1=np.sum(abs(ypredproba_all[1][:,0]-0.5))/ypredproba_all[1][:,0].shape[0]
ci2=np.sum(abs(ypredproba_all[2][:,0]-0.5))/ypredproba_all[2][:,0].shape[0]

ci0=ci1=ci2=1

finalcol[:,0]=(	ci0*(ypredproba_all[0][:,0] * ypredconfprob_all[0][0][0] + ypredproba_all[0][:,1] * ypredconfprob_all[0][0][1] ) +
				ci1*(ypredproba_all[1][:,1] * ypredconfprob_all[1][1][1] + ypredproba_all[1][:,0] * ypredconfprob_all[1][1][0] ) +
                ci2*(ypredproba_all[2][:,1] * ypredconfprob_all[2][1][1] + ypredproba_all[2][:,0] * ypredconfprob_all[2][1][0] ) )
				
finalcol[:,1]=( ci0*(ypredproba_all[0][:,1] * ypredconfprob_all[0][1][1] + ypredproba_all[0][:,0] * ypredconfprob_all[0][1][0] ) +
				ci1*(ypredproba_all[1][:,0] * ypredconfprob_all[1][0][0] + ypredproba_all[1][:,1] * ypredconfprob_all[1][1][0] ) +
				ci2*(ypredproba_all[2][:,1] * ypredconfprob_all[2][1][1] + ypredproba_all[2][:,0] * ypredconfprob_all[2][1][0] ) )
				
finalcol[:,2]=( ci0*(ypredproba_all[0][:,1] * ypredconfprob_all[0][1][1] + ypredproba_all[0][:,0] * ypredconfprob_all[0][1][0] ) + 
                ci1*(ypredproba_all[1][:,1] * ypredconfprob_all[1][1][1] + ypredproba_all[1][:,0] * ypredconfprob_all[1][1][0] ) +
                ci2*(ypredproba_all[2][:,0] * ypredconfprob_all[2][0][0] + ypredproba_all[2][:,1] * ypredconfprob_all[2][0][1] ) )
 	
finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))



indexs=[]
counter=0
for i in range(ypredproba_all[0].shape[0]):
    d=np.argmax(ypredproba_all[2][i])
    pd=np.argmax(ypredproba_all[1][i])
    n=np.argmax(ypredproba_all[0][i])
    
    if((d==0 and pd==1 and n==1) or (d==1 and pd==0 and n==1) or (d==1 and pd==1 and n==0) ):
        counter=counter+1
        indexs.append(i)
    
    
























