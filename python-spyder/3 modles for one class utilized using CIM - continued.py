# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 01:04:52 2020

@author: Henock
"""

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

data=datasets.load_wine()

x= data.data
y=data.target

xtest,xtrain,ytest_original,ytrain_original=train_test_split(x,y,random_state=randomseed,train_size=0.2)

clf=[]
acc=[]
ypredproba_all=[]
ypredconfprob_all=[]

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()




#=================================================
# classs 0
#=================================================
ytrain[ytrain!=0]=5
ytest[ytest!=0]=5

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

temp0=temp.copy()
temp=np.argmax(temp,axis=1)    

temp[temp==1]=5
temp[temp==0]=0

print(m.accuracy_score(ytest,temp))


#=======================================================================


#=================================================
# classs 1
#=================================================
clf=[]
acc=[]
ypredproba_all=[]
ypredconfprob_all=[]

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()

ytrain[ytrain!=1]=5
ytest[ytest!=1]=5


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

temp1=temp.copy()
temp=np.argmax(temp,axis=1)    

temp[temp==1]=5
temp[temp==0]=1

print(m.accuracy_score(ytest,temp))



#=================================================
# classs 2
#=================================================
clf=[]
acc=[]
ypredproba_all=[]
ypredconfprob_all=[]

ytrain=ytrain_original.copy()
ytest=ytest_original.copy()

ytrain[ytrain!=2]=5
ytest[ytest!=2]=5


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

temp2=temp.copy()
temp=np.argmax(temp,axis=1)    

temp[temp==1]=5
temp[temp==0]=2

print(m.accuracy_score(ytest,temp))



finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]=temp0[:,0]
finalcol[:,1]=temp1[:,0]
finalcol[:,2]=temp2[:,0]

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,np.argmax(finalcol,axis=1)))







