#!/usr/bin/env python
# coding: utf-8

# In[34]:


# Name : Cathaoir Agnew
# ID   : 16171659 

#2. Redo Question 1 using tournament selection and answer the same question but for an individual size of 100.


# individuals processed will be population size + population size * generation 

# Approx min individuals processed = 16,100

# below is just main-1.py , with slight changes to the code

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt

# problem constants: (individual of size 100)
ONE_MAX_LENGTH = 100  # length of bit string to be optimized

# Genetic Algorithm constants, which will vary to minimize size of total :
#POPULATION_SIZE = 1000 , MAX_GENERATIONS = 24 , hits prefect max score of 100 ,individuals processed = 25,000
#POPULATION_SIZE = 500 , MAX_GENERATIONS = 50 , doesnt hit prefect max score of 100
#POPULATION_SIZE = 750 , MAX_GENERATIONS = 21 , hits prefect max score of 100 , individuals processed = 16,500
#POPULATION_SIZE = 600 , MAX_GENERATIONS = 27 , doesnt hit prefect max score of 100
#POPULATION_SIZE = 700 , MAX_GENERATIONS = 22 , hits prefect max score of 100 , individuals processed = 16,100
#POPULATION_SIZE = 650 , MAX_GENERATIONS = 24 , doesnt hit prefect max score of 100
#POPULATION_SIZE = 675 , MAX_GENERATIONS = 22 , doesnt hit prefect max score of 100
#POPULATION_SIZE = 685 , MAX_GENERATIONS = 22 , doesnt hit prefect max score of 100
#POPULATION_SIZE = 690 , MAX_GENERATIONS = 22 , doesnt hit prefect max score of 100
#POPULATION_SIZE = 695 , MAX_GENERATIONS = 22 , doesnt hit prefect max score of 100

# As this task is just an approx
# estimate of the minimum individuals processed to find a perfect solution is given below:
# POPULATION_SIZE = 700 , MAX_GENERATIONS = 22 , hits prefect max score of 100 , individuals processed = 16,100


# concept for trial and error sorting:

# firstly big population size = 1000 and 30 generations got a max score of 100
# thought process was then to reduce population size, divide previous individuals processed size by population size to
# calculate how many generations I would be allowed to run to then process the same amount of individuals, and see what
# generation if any achieves the max score of 100. 

#Example of what I did:
# POPULATION_SIZE = 1000 , MAX_GENERATIONS = 30 , hits prefect max score of 100 , within 24 gens, individuals processed = 25,000
# reduce POPULATION_SIZE to 750 , (individuals processed) / (population size) = 25,000 / 750 = 33 
# so to minimize individuals processed; a population size of 750 must hit the max score of 100 within 32 generations, otherwise  
# 1000 population size and 24 generations is as good
# So running for population size = 750 , max generations = 33 , hits prefect max score of 100 within 21 generations 
# So population size =750 + 750 * 21 generations  =  individuals processed = 16,500

# Repeated this concept until I achieved a minimum individuals processed 


POPULATION_SIZE = 700
MAX_GENERATIONS = 22

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

# Tournament selection with tournament size of 3:
toolbox.register("select", tools.selTournament, tournsize= 3)

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
    
    print("Individuals processed (the population size multiplied by the number of generations required) = " , POPULATION_SIZE + (POPULATION_SIZE * MAX_GENERATIONS ) )

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




