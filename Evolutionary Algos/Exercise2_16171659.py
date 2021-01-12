#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name: Cathaoir Agnew
# ID:   16171659

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt


############################## hint from question 

# this is for creating items 
NBR_ITEMS = 100
MAX_WEIGHT = 1000

# set the random seed. Make sure you do this BEFORE creating the knapsack
# creating the knapsack
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Create the item dictionary: item name is an integer, and value is
# a (value, weight) 2-uple.

items = {}

# Create random items and store them in the items' dictionary.
for i in range(NBR_ITEMS):
     items[i] = (random.randint(1, 10), random.randint(1, 100))
            
####################################


# Genetic Algorithm constants:
POPULATION_SIZE = 500
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.05   # probability for mutating an individual
MAX_GENERATIONS = 750
HALL_OF_FAME_SIZE = 10

# set the random seed:
RANDOM_SEED = 50
random.seed(RANDOM_SEED)


toolbox = base.Toolbox()

# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)
#creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, NBR_ITEMS)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation:
def Knapsack(individual):
    
    weight = 0.0
    value = 0.0
    
    for i in range(len(individual)):
        if individual[i] == 1:
            value += items[i][0]
            weight += items[i][1]
   
    if weight > MAX_WEIGHT:
        return (MAX_WEIGHT-weight),                 # penalizing overweight bags        
    return value, 
        
toolbox.register("evaluate", Knapsack)

# genetic operators:

# toolbox.register("select", tools.selRoulette)

# Tournament selection with tournament size of 3:
toolbox.register("select", tools.selTournament, tournsize=3 )


# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/NBR_ITEMS)

# test


# Genetic Algorithm flow:
def main():
    
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # perform the Genetic Algorithm flow:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,
                                              stats=stats, halloffame=hof, verbose=True)


    # print Hall of Fame info:
    print("Hall of Fame Individuals = ", *hof.items, sep="\n")
    print("Best Ever Individual = ", hof.items[0])
    
        
    best = hof.items[0]
    weight_best = 0
    value_best = 0
    #declare weight variable
    #loop through best to calculate the weight
    #only add weight if its used, ie 1 , 
    #calc up weights
    
    
    for i in range(len(best)):
        if best[i] == 1:
            weight_best += items[i][1]
        else: 
            weight_best = weight_best
            
    print("Weight of best Individual: " , weight_best)
    
        
    for i in range(len(best)):
        if best[i] == 1:
            value_best += items[i][0]
        else: 
            value_best = value_best
            
    print("Value of best Individual: " , value_best)
                     
    #print("Weight of best Individual: " , best_ind_weight )
    
        
    # Genetic Algorithm is done - extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # Genetic Algorithm is done - plot statistics:
    #sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red' , label = 'Max')
    plt.plot(meanFitnessValues, color='green' , label = 'Avg')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.legend()
    plt.show()

 
if __name__ == '__main__':
    main()

