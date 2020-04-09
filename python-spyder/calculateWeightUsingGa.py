# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:33:59 2020

@author: Henock
"""

from pyeasyga import pyeasyga
import random
import numpy as np

np.random.seed(5)

# define a fitness function
def fitness(individual, data):
    
    # val= individual[0]*data[0]  +  individual[1]*data[1]  +  individual[2]*data[2] 
    val=0
    for i in range(len(individual)):
        val += individual[i]*data[i] 
    
    # # check the lowest value has the lowest weight    
    for i in range(len(individual)-1):
        if(individual[i]>individual[i+1]):
            val=0     
    # if val > 1:
    #     val=0   
                 
    return val

def create_individual(data):
    r = random.Random()
    # r.seed(5) 
    return list(np.sort([r.random() for _ in range(len(data))] ))

 
# def getbestvalues(data):    
    
#     temp_original=list(data.copy())
#     temp=list(np.sort(data))
#     temp_indexs=[]
    
#     for i in range (len(temp_original)):
#         temp_indexs.append(temp_original.index(temp[i]))
        
#     ga = pyeasyga.GeneticAlgorithm(data,
#                             population_size=50,
#                             generations=1000,
#                             crossover_probability=0.6,
#                             mutation_probability=0.02,
#                             elitism=True,
#                             maximise_fitness=True )
    
#     ga.data=temp
#     ga.create_individual = create_individual  
#     ga.fitness_function = fitness 

#     val=[]
#     best_individuals=[]
#     for i in range(5):  
#         ga.run()                                    # run the GA
#         print(i)
#         val.append(ga.best_individual()[0])   
#         best_individuals.append(ga.best_individual()[1])   
        
#     print(np.max(val),best_individuals[np.argmax(val)])  
    
#     finalresult=list(best_individuals[np.argmax(val)])
#     temp=np.zeros((len(finalresult),))
#     for i in range(len(finalresult)):
#         temp[temp_indexs[i]]=finalresult[i]
    
#     return temp

# # getbestvalues(data)


def getbestvalues(data):    

    # data=acc
    # data=[0.968,0.857,0.984,0.5,0.5]
    
    # data=[0.6863966818349385, 0.6852023910211884, 0.7335016228212894, 
    #       0.37570212754230187,0.7351489555824932,0.7048849042513139,0.7351489555824932] 
    #       # , , ]
        
    temp_original=list(data.copy())
    temp=list(np.sort(data))
    temp_indexs=[]
    
    i=0
    while i < (len(temp_original)):
        matchedindexs=np.where(np.array(temp_original) == temp[i])[0]    
        temp_indexs=temp_indexs+list(matchedindexs)
        i=i+matchedindexs.shape[0]
        
    ga = pyeasyga.GeneticAlgorithm(data,
                            population_size=500,
                            generations=100,
                            crossover_probability=0.6,
                            mutation_probability=0.02,
                            elitism=True,
                            maximise_fitness=True )
    
    ga.data=temp
    ga.create_individual = create_individual  
    ga.fitness_function = fitness 
    
    
    val=[]
    best_individuals=[]
    for i in range(5):  
        ga.data=temp
        ga.run()                                    # run the GA
        print(i)
        val.append(ga.best_individual()[0])   
        best_individuals.append(ga.best_individual()[1])   
        
    # print(np.max(val),best_individuals[np.argmax(val)])  
    
    finalresult=list(best_individuals[np.argmax(val)])
    finalresultordered=np.zeros((len(finalresult),))
    for i in range(len(finalresult)):
        finalresultordered[temp_indexs[i]]=finalresult[i]
    print(np.max(val),finalresultordered)
    
    return temp








