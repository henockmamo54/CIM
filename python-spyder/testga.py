# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:47:31 2020

@author: Henock
"""

# from pyeasyga.pyeasyga import GeneticAlgorithm
# import random
# import numpy as np

# np.random.seed(5)

# data = [0.1,2]
# ga = GeneticAlgorithm(data,
#                             population_size=20,
#                             generations=1000,
#                             crossover_probability=0.5,
#                             mutation_probability=0.05,
#                             elitism=True,
#                             maximise_fitness=False )




# # ga = GeneticAlgorithm(data)

# def fitness (individual, data):
#     fitness = 0
#     i=individual
#     fitness = pow(i[0],2) * data[0] + i[1]*data[1]
#     return fitness

# ga.fitness_function = fitness

# def create_individual(data):
#     return [random.random() for _ in range(len(data))]
# ga.create_individual = create_individual

# ga.run()
# print (ga.best_individual())



from pyeasyga import pyeasyga
import random
import numpy as np

# random.seed(5)


# setup data
data = [0.81,0.7,0.85]

# ga = pyeasyga.GeneticAlgorithm(data)        # initialise the GA with data
ga = pyeasyga.GeneticAlgorithm(data,
                            population_size=5,
                            generations=4,
                            crossover_probability=0.8,
                            mutation_probability=0.02,
                            elitism=True,
                            maximise_fitness=True )


# define a fitness function
def fitness(individual, data):
    
    val= individual[0]*data[0] +  individual[1]*data[1]+  individual[2]*data[2]
    
    # check the lowest value has the lowest weight
    if(np.argmin(individual)!=1):
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




# val=[]
# indexs=[]
# for i in range(10):  
#     ga.run()                                    # run the GA
#     val.append(ga.best_individual()[0])   
#     indexs.append(ga.best_individual()[1])   
    
# print(np.max(val),indexs[np.argmax(val)])    















