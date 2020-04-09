# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:40:32 2020

@author: Henock
"""
 
from pyeasyga import pyeasyga
import random
import numpy as np

np.random.seed(5)



# setup data
# data = [0.968,0.857,0.984]
# data=np.sort(data)

# ga = pyeasyga.GeneticAlgorithm(data)        # initialise the GA with data
# ga = pyeasyga.GeneticAlgorithm(data,
#                             population_size=50,
#                             generations=1000,
#                             crossover_probability=0.6,
#                             mutation_probability=0.02,
#                             elitism=True,
#                             maximise_fitness=True )


# define a fitness function
def fitness(individual, data):
    
    val= individual[0]*data[0]  +  individual[1]*data[1]  +  individual[2]*data[2]
    
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
    return [r.random() for _ in range(len(data))] 

# ga.create_individual = create_individual  
# ga.fitness_function = fitness               # set the GA's fitness function

# ga.run()                                    # run the GA
# print (ga.best_individual())                  # print the GA's best solution

 
# def getbestvalues_original(data):
#     ga.data=data
#     val=[]
#     indexs=[]
#     for i in range(10):  
#         ga.run()                                    # run the GA
#         print(i)
#         val.append(ga.best_individual()[0])   
#         indexs.append(ga.best_individual()[1])   
        
#     print(np.max(val),indexs[np.argmax(val)])  
 
def getbestvalues(data):    
    
    temp_original=list(data.copy())
    temp=list(np.sort(data))
    temp_indexs=[]
    
    for i in range (len(temp_original)):
        temp_indexs.append(temp_original.index(temp[i]))
        
    ga = pyeasyga.GeneticAlgorithm(data,
                            population_size=50,
                            generations=1000,
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
        ga.run()                                    # run the GA
        print(i)
        val.append(ga.best_individual()[0])   
        best_individuals.append(ga.best_individual()[1])   
        
    print(np.max(val),best_individuals[np.argmax(val)])  
    
    finalresult=list(best_individuals[np.argmax(val)])
    temp=np.zeros((len(finalresult),))
    for i in range(len(finalresult)):
        temp[temp_indexs[i]]=finalresult[i]
    
    return temp

# getbestvalues(data)




# data=acc

# temp_original=list(data.copy())
# temp=list(np.sort(data))
# temp_indexs=[]

# i=0
# while i < (len(temp_original)):
#     matchedindexs=np.where(np.array(temp_original) == temp[i])[0]
#     print(i,'matchedindexs',matchedindexs.shape[0],matchedindexs)
#     temp_indexs=temp_indexs+list(matchedindexs)
#     i=i+matchedindexs.shape[0]
    
# ga = pyeasyga.GeneticAlgorithm(data,
#                         population_size=50,
#                         generations=1000,
#                         crossover_probability=0.6,
#                         mutation_probability=0.02,
#                         elitism=True,
#                         maximise_fitness=True )

# ga.data=temp
# ga.create_individual = create_individual  
# ga.fitness_function = fitness 


# val=[]
# best_individuals=[]
# for i in range(5):  
#     ga.data=temp
#     ga.run()                                    # run the GA
#     print(i)
#     val.append(ga.best_individual()[0])   
#     best_individuals.append(ga.best_individual()[1])   
    
# print(np.max(val),best_individuals[np.argmax(val)])  

# finalresult=list(best_individuals[np.argmax(val)])
# temp=np.zeros((len(finalresult),))
# for i in range(len(finalresult)):
#     temp[temp_indexs[i]]=finalresult[i]








