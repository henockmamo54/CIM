# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:08:11 2020

@author: Henock
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:59:45 2020

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
svc=svmgpu(random_state=randomseed,probability=True,C=1)
svc.fit(xtrain,ytrain)

svcpred=svc.predict(xtest)
print(m.accuracy_score(ytest,svcpred))

clf.append(svc)
acc.append(m.accuracy_score(ytest,svcpred))
ypredproba_all.append(1-svc.predict_proba(xtest))

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
temp[:,0]=(ypredproba_all[0][:,0]*ypredconfprob_all[0][0][0] +    ypredproba_all[0][:,1]*ypredconfprob_all[0][0][1] +  
    ypredproba_all[1][:,0]*ypredconfprob_all[1][0][0] +       ypredproba_all[1][:,1]*ypredconfprob_all[1][0][1]  + 
    ypredproba_all[2][:,0]*ypredconfprob_all[2][0][0] +      ypredproba_all[2][:,1]*ypredconfprob_all[2][0][1] 
    )

temp[:,1]=(ypredproba_all[0][:,1]*ypredconfprob_all[0][1][1]+     ypredproba_all[0][:,0]*ypredconfprob_all[0][1][0] +        
     ypredproba_all[1][:,1]*ypredconfprob_all[1][1][1]+        ypredproba_all[1][:,0]*ypredconfprob_all[1][1][0]+        
    ypredproba_all[2][:,1]*ypredconfprob_all[2][1][1] + ypredproba_all[2][:,0]*ypredconfprob_all[2][1][0]
    )
	
temp0=temp.copy()/3
temp=np.argmax(temp,axis=1)    


temp[temp==1]=5
temp[temp==0]=0

temp0conf=m.confusion_matrix(ytest,temp)
temp0confh=np.sum(temp0conf,axis=1)

for i in range(temp0conf.shape[0]):
    temp0conf[i]= 100*temp0conf[i]/temp0confh[i] 
temp0conf=temp0conf/100

print('classs 0 ',m.accuracy_score(ytest,temp))
finalacc.append(m.accuracy_score(ytest,temp))



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
svc=svmgpu(random_state=randomseed,probability=True,C=1)
svc.fit(xtrain,ytrain)

svcpred=svc.predict(xtest)
print(m.accuracy_score(ytest,svcpred))

clf.append(svc)
acc.append(m.accuracy_score(ytest,svcpred))
ypredproba_all.append(1-svc.predict_proba(xtest))

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
temp[:,0]=(ypredproba_all[0][:,0]*ypredconfprob_all[0][0][0] +    ypredproba_all[0][:,1]*ypredconfprob_all[0][0][1] +  
    ypredproba_all[1][:,0]*ypredconfprob_all[1][0][0] +       ypredproba_all[1][:,1]*ypredconfprob_all[1][0][1]  + 
    ypredproba_all[2][:,0]*ypredconfprob_all[2][0][0] +      ypredproba_all[2][:,1]*ypredconfprob_all[2][0][1] 
    )

temp[:,1]=(ypredproba_all[0][:,1]*ypredconfprob_all[0][1][1]+     ypredproba_all[0][:,0]*ypredconfprob_all[0][1][0] +        
     ypredproba_all[1][:,1]*ypredconfprob_all[1][1][1]+        ypredproba_all[1][:,0]*ypredconfprob_all[1][1][0]+        
    ypredproba_all[2][:,1]*ypredconfprob_all[2][1][1] + ypredproba_all[2][:,0]*ypredconfprob_all[2][1][0]
    )

temp1=temp.copy()/3
temp=np.argmax(temp,axis=1)    

temp[temp==1]=5
temp[temp==0]=1

temp1conf=m.confusion_matrix(ytest,temp)
temp1confh=np.sum(temp1conf,axis=1)
for i in range(temp1conf.shape[0]):
    temp1conf[i]= 100*temp1conf[i]/temp1confh[i] 
temp1conf=temp1conf/100


print('classs 1 ',m.accuracy_score(ytest,temp))
finalacc.append(m.accuracy_score(ytest,temp))



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
svc=svmgpu(random_state=randomseed,probability=True,C=1)
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
temp[:,0]=(ypredproba_all[0][:,0]*ypredconfprob_all[0][0][0] +    ypredproba_all[0][:,1]*ypredconfprob_all[0][0][1] +  
    ypredproba_all[1][:,0]*ypredconfprob_all[1][0][0] +       ypredproba_all[1][:,1]*ypredconfprob_all[1][0][1]  + 
    ypredproba_all[2][:,0]*ypredconfprob_all[2][0][0] +      ypredproba_all[2][:,1]*ypredconfprob_all[2][0][1] 
    )

temp[:,1]=(ypredproba_all[0][:,1]*ypredconfprob_all[0][1][1]+     ypredproba_all[0][:,0]*ypredconfprob_all[0][1][0] +        
     ypredproba_all[1][:,1]*ypredconfprob_all[1][1][1]+        ypredproba_all[1][:,0]*ypredconfprob_all[1][1][0]+        
    ypredproba_all[2][:,1]*ypredconfprob_all[2][1][1] + ypredproba_all[2][:,0]*ypredconfprob_all[2][1][0]
    )

temp2=temp.copy()/3
temp=np.argmax(temp,axis=1)    


temp[temp==1]=5
temp[temp==0]=2

temp2conf=m.confusion_matrix(ytest,temp)
temp2confh=np.sum(temp2conf,axis=1)
for i in range(temp2conf.shape[0]):
    temp2conf[i]= 100*temp2conf[i]/temp2confh[i] 
temp2conf=temp2conf/100

print('classs 2 ',m.accuracy_score(ytest,temp))
finalacc.append(m.accuracy_score(ytest,temp))



finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]=temp0[:,0]*temp0conf[0][0] #+ temp0[:,0]*temp0conf[0][1]
finalcol[:,1]=temp1[:,0]*temp1conf[0][0] #+ temp0[:,0]*temp1conf[0][1]
finalcol[:,2]=temp2[:,0]*temp2conf[0][0] #+ temp0[:,0]*temp2conf[0][1]
finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))

#===========================================================================
#===========================================================================
#===========================================================================
  
finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]=finalacc[0]*(temp0[:,0]*temp0conf[0][0] - (temp0[:,1]*temp0conf[1][1] ))
finalcol[:,1]=finalacc[0]*(temp1[:,0]*temp1conf[0][0] - (temp0[:,1]*temp1conf[1][1] ))
finalcol[:,2]=finalacc[0]*(temp2[:,0]*temp2conf[0][0] - (temp0[:,1]*temp2conf[1][1] ))

finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))



#===========================================================================
#===========================================================================
#===========================================================================

finalcol=np.zeros((ytest.shape[0],3))
finalcol2=np.zeros((ytest.shape[0],3))

finalcol2[:,0]=(temp0[:,0])
finalcol2[:,1]=(temp1[:,0])
finalcol2[:,2]=(temp2[:,0])

finalcol[:,0]=finalacc[0]*(temp0[:,0])
finalcol[:,1]=finalacc[1]*(temp1[:,0])
finalcol[:,2]=finalacc[2]*(temp2[:,0])

finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))



#===========================================================================
#===========================================================================
#===========================================================================


finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]=(temp0[:,1]*temp0conf[1][1] + temp0[:,1]*temp0conf[1][0] )
finalcol[:,1]=(temp1[:,1]*temp1conf[1][1] + temp1[:,1]*temp0conf[1][0] )
finalcol[:,2]=(temp2[:,1]*temp2conf[1][1] + temp2[:,1]*temp0conf[1][0] )

finalpred=np.argmin(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))


#===========================================================================
#===========================================================================
#===========================================================================

temp0_e=temp0.copy()
temp0_e[:,0]=finalacc[0]*(temp0[:,0]  )
temp0_e[:,1]=finalacc[0]*(temp0[:,1]  )
									 
temp1_e=temp1.copy()                 
temp1_e[:,0]=finalacc[1]*(temp1[:,0]  )
temp1_e[:,1]=finalacc[1]*(temp1[:,1]  )
									 
temp2_e=temp2.copy()                 
temp2_e[:,0]=finalacc[2]*(temp2[:,0] )
temp2_e[:,1]=finalacc[2]*(temp2[:,1] ) 


finalcol=np.zeros((ytest.shape[0],3)) 
finalcol[:,0]=(temp0_e[:,0])
finalcol[:,1]=(temp1_e[:,0])
finalcol[:,2]=(temp2_e[:,0])

finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))



#===========================================================================
#===========================================================================
#===========================================================================
  
finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]=finalacc[0]*(temp0[:,0]*temp0conf[0][0] + (temp0[:,1]*temp0conf[0][1] ))
finalcol[:,1]=finalacc[0]*(temp1[:,0]*temp1conf[0][0] + (temp0[:,1]*temp1conf[0][1] ))
finalcol[:,2]=finalacc[0]*(temp2[:,0]*temp2conf[0][0] + (temp0[:,1]*temp2conf[0][1] ))

finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))



#===========================================================================
#===========================================================================
#===========================================================================
  

finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]=(temp0[:,1]*temp0conf[1][1] + temp0[:,1]*temp0conf[1][0] )
finalcol[:,1]=(temp1[:,1]*temp1conf[1][1] + temp1[:,1]*temp0conf[1][0] )
finalcol[:,2]=(temp2[:,1]*temp2conf[1][1] + temp2[:,1]*temp0conf[1][0] )

finalpred=np.argmin(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))





#===========================================================================
#===========================================================================
#===========================================================================
  

finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]=finalacc[0]*(temp0[:,0]*temp0conf[0][0] +    temp0[:,1]*temp0conf[0][1] )/np.sum(finalacc)
finalcol[:,1]=finalacc[1]*(temp1[:,0]*temp1conf[0][0] +    temp1[:,1]*temp1conf[0][1])/np.sum(finalacc)
finalcol[:,2]=finalacc[2]*(temp2[:,0]*temp2conf[0][0] +    temp2[:,1]*temp2conf[0][1])/np.sum(finalacc)

finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))







#===========================================================================
#===========================================================================
#=========================================================================== 

finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]=finalacc[0]*(temp0[:,1]*temp0conf[1][1] + temp0[:,1]*temp0conf[1][0] )
finalcol[:,1]=finalacc[0]*(temp1[:,1]*temp1conf[1][1] + temp1[:,1]*temp0conf[1][0] )
finalcol[:,2]=finalacc[0]*(temp2[:,1]*temp2conf[1][1] + temp2[:,1]*temp0conf[1][0] )

finalpred=np.argmin(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))


#===========================================================================
#===========================================================================
#===========================================================================
  
finalcol=np.zeros((ytest.shape[0],3))
finalcol[:,0]= (temp0[:,0] )
finalcol[:,1]= (temp1[:,0] )
finalcol[:,2]= (temp2[:,0] )

finalpred=np.argmax(finalcol,axis=1)

ytest=ytest_original.copy()
print(m.accuracy_score(ytest,finalpred))
print( m.confusion_matrix(ytest,finalpred))
































