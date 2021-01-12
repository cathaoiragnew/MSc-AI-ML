#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name : Cathaoir Agnew
# ID   : 16171659 

#3. Create your own fitness function works similar to OneMax 
#   except that the maximum fitness is when 50% of the individual's genes are 1, with a linear decay on either side

#  Approx min individuals processed = 7 , it was actually found in the parents 

#  Q. Why is this problem so much easier than OneMax?

#  A. We are making it easier to achieve higher in the halfmax fitness function, the One Max problem requires all of the
# individual genes to be of value 1, while our new fitness function only requires 50% of the individual genes
# to have a value of 1, as the values have a 50/50 chance of being 0/1 , there is a higher probability that we would see
# an individual with 50% of its genes with a value = 1 

# Probability for say a string length of 10 for all genes values = 1 , 
# as there is only 1 way for this to happen P( all 1 ) = (1/2)^10

# Probability for say a string length of 10 to have half genes values = 1  is (1/2)^10 
# now this is where its different, we have 10 Pick 5 ways of selecting five 1's and five  0's 
# 10 P 5 = 30,240  , so the equation becomes 30,240 * ((1/2)^10)

#  we can see for a string length of l for all genes to be = 1  is (1/2)^l
#  we can see for say a string length of l to have at least half of the values = 1  
# for a length l       l P (l/2) * ((1/2)^(l))


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

#POPULATION_SIZE = 10 , MAX_GENERATIONS = 0 , hits prefect max score of 50 ,individuals processed = 10
#POPULATION_SIZE = 5 , MAX_GENERATIONS = 2 , doesnt hit prefect max score of 100
#POPULATION_SIZE = 7 , MAX_GENERATIONS = 0 , hits prefect max score of 50 ,individuals processed = 7
#POPULATION_SIZE = 6 , MAX_GENERATIONS = 0 , doesnt hit prefect max score of 50
#POPULATION_SIZE = 3 , MAX_GENERATIONS = 1 , doesnt hit prefect max score of 50
#POPULATION_SIZE = 2 , MAX_GENERATIONS = 2 , doesnt hit prefect max score of 50
#POPULATION_SIZE = 1 , MAX_GENERATIONS = 6 , doesnt hit prefect max score of 50


POPULATION_SIZE = 7
MAX_GENERATIONS = 0

# Genetic Algorithm constants, which remain constant:
P_CROSSOVER = 0.8  # probability for crossover
P_MUTATION = 0.01   # probability for mutating an individual
HALL_OF_FAME_SIZE = 5

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


# we are going to change this code, as we are changing the fitness function:

# fitness calculation:
# compute the number of '1's in the individual
# check if there is at least 50% of 1's in the individual total genes

# example for a string of size 10
# max fitness value will 5, as 10/2 = 5 
# so half way is 5 . => 10 - 5 = 5    (symmetry)
# => max fitness values will always be ONE_MAX_LENTH / 2 , if a solution that contains 50% of genes = 1 

def halfMaxFitness(individual):
    """When a max value of ONE_MAX_LEGNTH/2 is returned, there will exist at least one soulution that contains half of the 
       genes equal to a value of 1, as this is what the maths below ensures as it's symmetric."""
    i = sum(individual)
    if i <= (len(individual)/2):
        return (i),
    else:
        return (len(individual)-i),

toolbox.register("evaluate", halfMaxFitness)

# genetic operators:

# Tournament selection with tournament size of 3:
toolbox.register("select", tools.selTournament, tournsize=3 )

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

    #print Hall of Fame info:
    print("Hall of Fame Individuals = ", *hof.items, sep="\n")
    print("Best Ever Individual = ", hof.items[0])
    
    print("Individuals processed (the population size multiplied by the number of generations required) = " ,POPULATION_SIZE + (POPULATION_SIZE * MAX_GENERATIONS) )

    # Genetic Algorithm is done - extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # Genetic Algorithm is done - plot statistics:
    #sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red' , label = 'Max')
    plt.plot(meanFitnessValues, color='green' , label = 'Mean')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.legend(loc='upper left' , fontsize = "large")
    plt.title('Max and Average Fitness over Generations')
    plt.show()

 
if __name__ == '__main__':
    main()


# In[ ]:




