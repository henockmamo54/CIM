# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:57:00 2020

@author: Henock
"""

from sklearn import datasets
from sklearn import metrics as m
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score 
import numpy as np
from itertools import combinations

randomseed=5
np.random.seed(randomseed)


comb1 = (list(combinations([0,1, 2, 3], 2) ))
comb2 = (list(combinations([0,1, 2, 3], 2) ))
comb3 = (list(combinations([0,1, 2, 3], 3) ))
comb4 = (list(combinations([0,1, 2, 3], 4) ))

comb=comb1+comb2+comb3+comb4 
comb=np.array(comb[14:20])

iris=datasets.load_iris()

x= iris.data
y=iris.target
xtest,xtrain,ytest,ytrain=train_test_split(x,y,random_state=randomseed,train_size=0.2)

clf=[]
acc=[]
ypredproba_all=[]
ypredconfprob_all=[]


for i in range(comb.shape[0]):    
    
    print(i,'**************************\\\\')
    
    rfc=RandomForestClassifier(n_estimators=100,random_state=randomseed)
    rfc.fit(xtrain[:,comb[i]],ytrain)
    
    ypred=rfc.predict(xtest[:,comb[i]])
    ypredprob=rfc.predict_proba(xtest[:,comb[i]])
    
    ypredproba_all.append(ypredprob)    
    acc.append(m.accuracy_score(ytest,ypred))     
#    print('accuracy = ', m.accuracy_score(ytest,ypred)) 
    
    confmat=m.confusion_matrix(ytest,ypred)
    confsumh=np.sum(confmat,axis=1)
    print(confmat)
    propconfmat=confmat.copy()
    
    print('==============>>><<<')
    for i in range(propconfmat.shape[0]):
        propconfmat[i]= 100*propconfmat[i]/confsumh[i]  
    print(propconfmat)
    ypredconfprob_all.append(propconfmat/100)


#probability being class 0
pbc0=[]
c=0
for i in range(xtest.shape[0]):
    val=0
    for j in range(comb.shape[0]): # classifiers
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][c]
    pbc0.append(val)
        

#probability being class 1
pbc1=[]
c=1
for i in range(xtest.shape[0]):
    val=0
    for j in range(comb.shape[0]): # classifiers
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][c]
    pbc1.append(val)
        

#probability being class 2
pbc2=[]
c=2
for i in range(xtest.shape[0]): 
    val=0
    for j in range(comb.shape[0]): # classifiers
        val+=ypredproba_all[j][i][c]*ypredconfprob_all[j][c][c]
    pbc2.append(val)
        

temp=np.zeros((len(pbc0),3))
temp[:,0]=pbc0
temp[:,1]=pbc1
temp[:,2]=pbc2
        
print(m.accuracy_score(ytest,np.argmax(temp,axis=1)))
    


    



