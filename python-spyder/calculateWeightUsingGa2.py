# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:12:36 2020

@author: Henock
"""
 

from pyeasyga import pyeasyga
import random
import numpy as np

np.random.seed(5)

# define a fitness function
def fitness(individual, data):
     
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
 

def getbestvalues(data):    
        
    temp_original=list(data.copy())
    temp=list(np.sort(data))
    temp_indexs=[]
    
    i=0
    while i < (len(temp_original)):
        matchedindexs=np.where(np.array(temp_original) == temp[i])[0]    
        temp_indexs=temp_indexs+list(matchedindexs)
        i=i+matchedindexs.shape[0]
        
    ga = pyeasyga.GeneticAlgorithm(data,
                            population_size=200,
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
    for i in range(10):  
        ga.data=temp
        ga.run()                                    # run the GA
        print(i)
        val.append(ga.best_individual()[0])   
        best_individuals.append(ga.best_individual()[1])   
            
    finalresult=list(best_individuals[np.argmax(val)])
    finalresultordered=np.zeros((len(finalresult),))
    for i in range(len(finalresult)):
        finalresultordered[temp_indexs[i]]=finalresult[i]
    print(np.max(val),finalresultordered)
    
    return finalresultordered








