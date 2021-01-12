#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Week 6

# Question 3: Salustowicz 
# Name: Cathaoir Agnew
# ID :  16171659


import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


import matplotlib.pyplot as plt

# Genetic Programming constants:
POPULATION_SIZE = 20
P_CROSSOVER = 0.7  # probability for crossover
P_MUTATION = 0.01   # probability for mutating an individual
MAX_GENERATIONS = 10
HALL_OF_FAME_SIZE = 10

# set the random seed:
RANDOM_SEED = 4
random.seed(RANDOM_SEED)

N_RUNS = 30

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
def psin(n):
    try:
        return numpy.sin(n)
    except Exception:
        return numpy.nan
    

def pcos(n):
    try:
        return numpy.cos(n)
    except Exception:
        return numpy.nan
    

def pow2(n):
    return operator.pow(n, 2)

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(psin, 1)
pset.addPrimitive(pcos, 1)
pset.addPrimitive(pow2, 1)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(math.sin, 1)
# moved this line into main function to avoid an error
pset.renameArguments(ARG0='x')


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the Salustowicz function :
    sqerrors = (  (func(x) -  (math.exp(-x) * (x**3) * pcos(x) * psin(x) ) * ( pcos(x) * (pow2(psin(x))) - 1 )  )**2  for x in points)
    myError = math.fsum(sqerrors) / len(points)
    if myError>1000:
        myError=1000
    return myError,

toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-5,5)])
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    
    minFitList = []
    avgFitList = []
    avgSizeList = []
    stdSizeList = []

    for i in range(0,N_RUNS):
        
        # this is moved line, as was generating error otherwise
        pset.addEphemeralConstant("rand101" + str(random.random()) , lambda: random.randint(-1,1))


        population = toolbox.population(n=POPULATION_SIZE)
        # define the hall-of-fame object:
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)   

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                       ngen=MAX_GENERATIONS, stats=mstats,
                                       halloffame=hof, verbose=True)

        minFitnessValues, meanFitnessValues = logbook.chapters['fitness'].select("min", "avg")
        avgSize, stdSize = logbook.chapters['size'].select("avg", 'std')

        # Save statistics for this run:
        minFitList.append(minFitnessValues)
        avgFitList.append(meanFitnessValues)
        
        avgSizeList.append(avgSize)
        stdSizeList.append(stdSize)

        #Genetic Programming is done with this run - plot statistics:
        #plt.plot(maxFitnessValues, color='red')
        #plt.plot(meanFitnessValues, color='green')
        #plt.xlabel('Generation')
        #plt.ylabel('Max / Average Fitness')
        #plt.title('Max and Average Fitness over Generations')
        #plt.show()
        
    # Genetic Algorithm is done (all runs) - plot statistics:
    x = numpy.arange(0, MAX_GENERATIONS+1)
    
    avgArray = numpy.array(avgFitList)
    minArray = numpy.array(minFitList)
    treeSizeArray = numpy.array(avgSizeList)
    stdSizeArray = numpy.array(stdSizeList)
    
    plt.errorbar(x,avgArray.mean(0))
    plt.errorbar(x,minArray.mean(0))
    plt.show()
    
    
    plt.errorbar(x, treeSizeArray.mean(0) , yerr = stdSizeArray.mean(0))
    plt.xlabel('Generation')
    plt.ylabel("Average number of trees")
    plt.title("Graph of average size of trees")
    plt.show()
    
    #plot above accounts for std 
    
    ####################################################################
    
    # plot below is more clean looking without std
    
    plt.errorbar(x, treeSizeArray.mean(0))
    plt.xlabel('Generation')
    plt.ylabel("Average number of trees")
    plt.title("Graph of average size of trees")
    plt.show()

if __name__ == "__main__":
    main()

