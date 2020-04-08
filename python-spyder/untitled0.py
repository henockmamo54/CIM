# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:26:20 2020

@author: Henock
"""



from scipy.stats import entropy
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

data=datasets.load_breast_cancer()
x=data.data
y=data.target



# print(entropy([1/2, 1/2], base=2))


print(entropy( [0.4 , 0.6 ], base=2))




# print(entropy(np.arange(0,10), base=2))
# print(entropy([5,5,5,5], base=2))

# for i in range(x.shape[1]): 
#     print(i,entropy(x[:,i], base=2))
    
    
    
# print( 1/(entropy([0.99, 0.01, 0.99, 0.01], base=2)))
# print(  1/(entropy([0.5, 0.5, 0.5, 0.5], base=2)))


# print(1/entropy([0.5, 0.5, 0.5, 0.5]))


    
# print( (entropy([0.99, 0.01 ], base=2)))
# # print(  1/(entropy([0.5, 0.5 ], base=2)))

temp=np.arange(0,1,0.1)
entropyval=[]

for i in range(len(temp)):
    entropyval.append(1/entropy([temp[i],1-temp[i]+0.5],base=2))
    
plt.plot(temp,entropyval)
plt.plot(1-temp,entropyval)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
 

ax.view_init(45, angle)
ax.scatter(temp, 1-temp, entropyval)


ax.view_init(30, angle)
ax.scatter(temp, 1-temp, entropyval)


ax.view_init(60, angle)
ax.scatter(temp, 1-temp, entropyval)


for angle in range(0, 360):        
    ax.view_init(60, angle)
    ax.scatter(temp, 1-temp, entropyval)



