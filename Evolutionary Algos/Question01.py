#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name : Cathaoir Agnew
# ID   : 16171659 

# 1. Using roulette wheel selection, what is the smallest population size and number of generations
# you can you find a perfect solution with an individual size of 50?  
# Give your answer in terms of individuals processed, that is,
# the population size multiplied by the number of generations required.

# individuals processed will be population size + population size * generation 

# Approx min individuals processed = 34,840


# below is just main-1.py , with slight changes to the code

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt

# problem constants: (individual of size 50)
ONE_MAX_LENGTH = 50  # length of bit string to be optimized

# Genetic Algorithm constants, which will vary to minimize size of total :
#POPULATION_SIZE = 1000 , MAX_GENERATIONS = 58 , hits prefect max score of 50 ,individuals processed = 59,000
#POPULATION_SIZE = 500 , MAX_GENERATIONS = 113 , doesnt hit prefect max score of 50
#POPULATION_SIZE = 750 , MAX_GENERATIONS = 73 , hits prefect max score of 50 ,individuals processed = 55,500
#POPULATION_SIZE = 600 , MAX_GENERATIONS = 91 , hits prefect max score of 50 ,individuals processed = 55,200
#POPULATION_SIZE = 550 , MAX_GENERATIONS = 91 , hits prefect max score of 50 ,individuals processed = 50,600
#POPULATION_SIZE = 525 , MAX_GENERATIONS = 72,  hits prefect max score of 50 ,individuals processed = 38,325
#POPULATION_SIZE = 510 , MAX_GENERATIONS = 75,  doesnt hit prefect max score of 50
#POPULATION_SIZE = 520 , MAX_GENERATIONS = 66 , hits prefect max score of 50 ,individuals processed = 34,840 

# As this task is just an approx
# estimate of the minimum individuals processed to find a perfect solution is given below:
#POPULATION_SIZE = 520 , MAX_GENERATIONS = 66 , hits prefect max score of 50 ,individuals processed = 34,840

# concept for trial and error sorting:

# firstly big population size = 1000 and 60 generations got a max score of 50 , after 58 generations
# thought process was then to reduce population size, divide previous individuals processed size by population size to
# calculate how many generations I would be allowed to run to then process the same amount of individuals, and see what
# generation if any achieves the max score of 50. 

#Example of what I did:
# POPULATION_SIZE = 1000 , MAX_GENERATIONS = 58 , hits prefect max score of 50 , individuals processed = 59,000
# reduce POPULATION_SIZE to 750 , (individuals processed) / (population size) = 59,000 / 750 = 78 
# so to minimize individuals processed; a population size of 750 must hit the max score of 50 within 78 generations, otherwise  
# 1000 population size and 58 generations is as good
# So running for population size = 750 , max generations = 78 , hits prefect max score of 50 within 73 generations 
# So population size = 750 + 750 * 73 generations  =  individuals processed = 55,500

# Repeated this concept until I achieved a minimum individuals processed 


POPULATION_SIZE = 520
MAX_GENERATIONS = 66

# Genetic Algorithm constants, which remain constant:
P_CROSSOVER = 0.8  # probability for crossover
P_MUTATION = 0.01   # probability for mutating an individual
HALL_OF_FAME_SIZE = 10

# set the random seed:
RANDOM_SEED = 42
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
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation:
# compute the number of '1's in the individual
def oneMaxFitness(individual):
    return sum(individual),  # return a tuple


toolbox.register("evaluate", oneMaxFitness)

# genetic operators:

# Roulette wheel selection 
toolbox.register("select", tools.selRoulette)


# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)


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
    
    print("Individuals processed (the population size multiplied by the number of generations required) = " ,POPULATION_SIZE + (POPULATION_SIZE * MAX_GENERATIONS) )

    # Genetic Algorithm is done - extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # Genetic Algorithm is done - plot statistics:
    #sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()

 
if __name__ == '__main__':
    main()


# In[ ]:




