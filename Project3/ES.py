#!/usr/local/bin/python3

"""
Author:     Clint Cooper, Emily Rohrbough, Leah Thompson
Date:       11/01/15
CSCI 447:   Project 3

Code for training a Neural Network via a Evolution Strategy
For more accurate results, run on more training sets.
Input is in the format of:
([Input Vectors of Values], [Output Vectors of Values], size of the population
used, number of participants in the tournaments, number of victors to be used
as parents, number of generations to iterate through, Threshold,
Crossover Rate, Mutation Rate)
The returned structure is a NN that has been trained via the ES.
"""

import GA
import random
import numpy
import NN
import copy
from operator import itemgetter
import math

OrigAnswers = []
hero = 0

def init_pop(es_net, inputs, outputs, size):
    '''ES citizens have 2 vectors: x = reals, sigma = strategies <x, sigma'''
    pop = []
    pop = GA.generatePopulation(es_net, inputs, outputs, size)
    for i in range(len(pop)):
        del pop[i][-1]
        for j in range(len(pop[i])):
            pop[i].append(numpy.std(pop[i]))
        pop[i].append(0)
    return pop


def gauss():
    '''Random Gaussian N(0, 1)'''
    #        w = 1
    #        while w >= 1:
    #            mu_1 = 2*random.random() - 1
    #            mu_2 = 2*random.random() - 1
    #            w = mu_1*mu_2 + mu_2*mu_2
    #        w = math.sqrt((-2.0*numpy.log(w))/w)
    return numpy.random.normal(0, 1)  # (mu_2*w)

# mutate the real-values
def mutate_vec(child):
    '''Mutate the real valued vector bassed off of strategy params'''
    length = int((len(child) - 1) / 2)
    for i in range(length):
        child[i] = (child[i] + child[length + i] * gauss())
#            x[i] = parents[i][0] if x[i] < parents[i][0]
#            x[i] = parents[i][1] if x[i] > parents[i][1]


def mutate_strat(child):
    '''Get std devs for a child, x', pg 218'''
    length = int((len(child) - 1) / 2)
    t = math.sqrt(2.0 * length)**(-1.0)
    t_prime = math.sqrt(2.0 * (math.sqrt(length)))**(-1.0)
    for i in range(length, (len(child) - 1)):
        child[i] = (child[i] * numpy.exp(t_prime * gauss() + t * gauss()))


def mutate(child, mRate):
    '''This mutates the child.'''
    length = int((len(child) - 1)/2)
    for i in range(length):
        if random.random() < mRate:
            mutate_vec(child)
            mutate_strat(child)

def train(inputs, outputs, size, participants, victors, generations, threshold, cRate, mRate, printFile=False):
    '''Create and start training the NN via evolution strategy. Selection, crossover, mutation, evaluation.''' 
    global hero
    global OrigAnswers
    OrigAnswers = copy.deepcopy(outputs)
    EvaluationNN = GA.create_net(inputs, outputs)
    population = init_pop(EvaluationNN, inputs, outputs, size)
    # Test each citizen and determine initial fitness
    GA.evaluate(EvaluationNN, population, inputs, outputs)

    if printFile: f = open('ES.csv', 'w')
    gen = 0
    children = []
    # loop until a hero is found or we've reached max generations
    while gen <= generations and hero == 0:
        # Select our parents using tournament selection
        parents = GA.tournament(population, participants, victors)
        # Have our parents mate (Crossover)
        children = GA.mate(parents, cRate)
        # Have the children experience the world (Mutate)
        for child in children:
            mutate(child, mRate)
        # Test each child's fitness
        GA.evaluate(EvaluationNN, children, inputs, outputs)
        children = GA.tournament(children, participants, victors)
        population = sorted(population + children,
                            key=itemgetter(-1))[:-victors]
        if GA.heroFound(population, threshold):
            break
        else:
            print("Training: {:2.2%}".format(
                population[0][-1]), "{:2.2%}     ".format(gen / generations), end="\r")
            if printFile: f.write('%f,' % population[0][-1])
            if printFile: f.write('\n')
        gen += 1
    if printFile: f.close()
    if hero == 0:
        gen -= 1
        hero = sorted(population, key=itemgetter(-1))[0]
    EvaluationNN.SetNNWeights(hero[:-1])  # Load hero into NN, prep for usage.

    # Evaluate the hero on the inputs and outputs
    print('Generations: %d' % gen, ' ' * 20)
    print("Error Relative: {:2.5%}".format(NN.calcRelativeError(EvaluationNN, inputs, OrigAnswers)))
    print("Least Squares: %d" % NN.calcLeastSquaresError(EvaluationNN, inputs, OrigAnswers))
    print("Loss Squared: %d" % NN.calcLossSquared(EvaluationNN, inputs, OrigAnswers))
    #for x in inputs:
    #    EvaluationNN.SetStartingNodesValues(x)
    #    EvaluationNN.CalculateNNOutputs()
    #    print(x, EvaluationNN.GetNNResults(), EvaluationNN.GetNNResultsInt(), OrigAnswers[inputs.index(x)])
    print()

    return EvaluationNN


def main(inputs, outputs, size=20, members=10, victors=5, generations=100,
         threshold=5, cRate=0.2, mRate=0.2, printFile=False):
    global OrigAnswer
    OrigAnswers = []
    global hero
    hero = 0
    es_nn = train(inputs, outputs, size, members, victors, generations, threshold,
                  cRate, mRate)


if __name__ == '__main__':
    print('Starting some ES training...\n')
    #main([[2, 3], [1, 3], [3, 3]], [[101], [400], [3604]], size=20,
    #     members=10, victors=5, generations=100000, threshold=5, cRate=0.25, mRate=0.75, printFile=False)
    main([[1,1,1,1], [1,0,1,0], [0,0,1,1]], [[2],[1],[1]], size=20, members=10, 
            victors=5, generations=100000, threshold=5, cRate=0.75, mRate=0.25, printFile=False)
