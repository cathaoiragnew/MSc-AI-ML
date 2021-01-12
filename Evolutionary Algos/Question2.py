#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Week 6 

# Q2. Experiment with population sizes to solve the 6 Even Parity Problem;
# report on the smallest number of evaluations that you were able to achieve success with,
# that is, your population size * number of generationss.

# approx smallest number of evaluations: 17,500    achieved with a population size of 1750 & max gens = 9

# Even parity fitness is achieved when there is an odd number of 1's in the bit string
# 1 if there is odd number of 1's in bit string 
# 0 if there is odd number of 1's in bit string

# since the fitness function is actually getCost,
# we want to minize partiy error, so if a fitness value of 0 is returned we have found a solution


# Genetic Algorithm constants, which will vary to minimize size of total :
#POPULATION_SIZE = 2000 , MAX_GENERATIONS = 31 , hits perfect min score of 0 ,individuals processed = 64,000
#POPULATION_SIZE = 1000 , MAX_GENERATIONS = 64 , doesnt hit  min score of 0
#POPULATION_SIZE = 1500 , MAX_GENERATIONS = 42 , doesnt hit  min score of 0
#POPULATION_SIZE = 1750 , MAX_GENERATIONS = 9 , hits prefect max score of 50 ,individuals processed = 17,500
 

import random
import operator

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib.pyplot as plt
import itertools
import networkx as nx


# Genetic Algorithm constants:
POPULATION_SIZE = 1750
P_CROSSOVER = 0.9
P_MUTATION = 0.01
MAX_GENERATIONS = 9
HALL_OF_FAME_SIZE = 10

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# problem constants:
NUM_INPUTS = 6
NUM_COMBINATIONS = 2 ** NUM_INPUTS

# Genetic Programming specific constants:
MIN_TREE_HEIGHT = 3
MAX_TREE_HEIGHT = 5
LIMIT_TREE_HEIGHT = 17
MUT_MIN_TREE_HEIGHT = 0
MUT_MAX_TREE_HEIGHT = 2

toolbox = base.Toolbox()

# calculate the truth table of even parity check:
parityIn = list(itertools.product([0, 1], repeat=NUM_INPUTS))
parityOut = []
for row in parityIn:
    parityOut.append(sum(row) % 2)

# create the primitive set:
primitiveSet = gp.PrimitiveSet("main", NUM_INPUTS, "in_")
primitiveSet.addPrimitive(operator.and_, 2)
primitiveSet.addPrimitive(operator.or_, 2)
primitiveSet.addPrimitive(operator.xor, 2)
primitiveSet.addPrimitive(operator.not_, 1)

# add terminal values:
primitiveSet.addTerminal(1)
primitiveSet.addTerminal(0)

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on the primitive tree:
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# create a helper function for creating random trees using the primitive set:
toolbox.register("expr", gp.genFull, pset=primitiveSet, min_=MIN_TREE_HEIGHT, max_=MAX_TREE_HEIGHT)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.expr)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# create an operator to compile the primitive tree into python code:
toolbox.register("compile", gp.compile, pset=primitiveSet)

# calculate the difference between the results of the
# generated function and the expected parity results:
def parityError(individual):
    func = toolbox.compile(expr=individual)
    return sum(func(*pIn) != pOut for pIn, pOut in zip(parityIn, parityOut))

# fitness measure:
def getCost(individual):
    return parityError(individual), # return a tuple

toolbox.register("evaluate", getCost)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=MUT_MIN_TREE_HEIGHT, max_=MUT_MAX_TREE_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitiveSet)

# bloat control:
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT_TREE_HEIGHT))

# Genetic Algorithm flow:
def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)


    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # perform the Genetic Algorithm flow
    population, logbook = algorithms.eaSimple(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=mstats,
                                                      halloffame=hof,
                                                      verbose=True)


    maxFitnessValues, meanFitnessValues = logbook.chapters['fitness'].select("min", "avg")


    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- length={}, height={}".format(len(best), best.height))
    print("-- Best Fitness = ", best.fitness.values[0])
    print("Number of individuals processed: ", POPULATION_SIZE + POPULATION_SIZE*MAX_GENERATIONS )

    #Genetic Programming is done with this run - plot statistics:
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()

