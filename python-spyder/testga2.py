# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:41:39 2020

@author: Henock
"""
 

from pyeasyga import pyeasyga
import random
import numpy as np

# random.seed(5)♦


# setup data
data = [0.968,0.857,0.984]
data=np.sort(data)

# ga = pyeasyga.GeneticAlgorithm(data)        # initialise the GA with data
ga = pyeasyga.GeneticAlgorithm(data,
                            population_size=50,
                            generations=1000,
                            crossover_probability=0.8,
                            mutation_probability=0.02,
                            elitism=True,
                            maximise_fitness=True )


# define a fitness function
def fitness(individual, data):
    
    val= individual[0]*data[0]  +  individual[1]*data[1]  +  individual[2]*data[2]
    
    # # check the lowest value has the lowest weight
    
    # if(not (individual[2]>individual[1] and individual[2]>individual[0] and individual[0]>individual[1]  )):
    #     val=0
    
    for i in range(len(individual)-1):
        if(individual[i]>individual[i+1]):
            val=0        
    
    if val > 1:
        val=0   
        
    # print(individual)
    # print(val,individual[0]*data[0] +  individual[1]*data[1]+  individual[2]*data[2])
        
    return val

def create_individual(data):
    return [random.random() for _ in range(len(data))]
ga.create_individual = create_individual


ga.fitness_function = fitness               # set the GA's fitness function
ga.run()                                    # run the GA
print (ga.best_individual())                  # print the GA's best solution




val=[]
indexs=[]
for i in range(10):  
    ga.run()                                    # run the GA
    print(i)
    val.append(ga.best_individual()[0])   
    indexs.append(ga.best_individual()[1])   
    
print(np.max(val),indexs[np.argmax(val)])    















