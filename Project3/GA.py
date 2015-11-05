#!/usr/local/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	11/01/15
CSCI 447:	Project 3

Code for training a Neural Network via a genetic algorithm - template for ES/DE
For more accurate results, run on more training sets.
Input is in the format of:
([Input Vectors of Values], [Output Vectors of Values], size of the population
used, number of participants in the tournaments, number of victors to be used
as parents, number of generations to iterate through, Threshold,
Crossover Rate, Mutation Rate)
The returned structure is a NN that has been trained.
"""

import NN
import random
import copy
from operator import itemgetter

OrigAnswers = []
hero = 0


def generatePopulation(net, inputs, outputs, size):
    '''Create citizens as arrays of weights that will be injected and stripped
    from the NN'''
    citizenTemp = net.GetNNWeights()
    population = []
    for i in range(size):
        for j in range(len(citizenTemp)):
            # Random weights for the NN topology
            citizenTemp[j] = (random.random() * 1) - 0.5
        population.append(copy.deepcopy(citizenTemp))
        # Each citizen tracks their current fitness based on the dimensionality
        # of the outputs
        population[-1].append(0)
    return population


def crossover(parent1, parent2, rate=0.2):
    '''2 parents 'mate' and produce a child where a random number of crossovers
    have occured at random locations based on a rate percentage'''
    child = []
    current = 0
    for i in range(len(parent1[:-1])):
        if random.random() < rate:
            current = (current + 1) % 2
        if current == 0:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    child.append(0)  # child has not yet tested
    return child


def mutate(child, rate=0.2):
    '''random number of mutations that occur on random weights based on a rate
    percentage'''
    for i in range(len(child[:-1])):
        # chance child will experience something that enlightens them
        if random.random() < rate:
            child[i] += (random.random() * 1) - 0.5


def tournament(population, participants, victors):
    '''Citizens are selected via random distribution and ranked based on
    fitness where we'll pull some number of potential parents via elitism.
    We select some number of participants randomly from the population and
    rank them, then selecting victors number of them as return set.'''
    bracket = sorted(random.sample(
        population, participants), key=itemgetter(-1))
    return bracket[0:victors]


def evaluate(NNWorking, population, inputs, outputs):
    '''Tests each citizen in the population against a NN topology with inputs
    and outputs to generate an cumulitive fitness measurement, which should be
    minimized'''
    for citizen in population:
        citizen[-1] = 0
        NNWorking.SetNNWeights(citizen[:-1])  # Load weights into the NN
        for i in range(len(inputs)):
            NNWorking.SetStartingNodesValues(inputs[i])  # Load inputs into NN
            NNWorking.CalculateNNOutputs()  # Run the NN once
            # Calculate the fitness value and let the citizen track it
            # for j in range(len(NN.GetNNResults())):
            #    citizen[-1] += ((outputs[i][j] - NN.GetNNResults()[j]))**2
            citizen[-1] += NN.calcRelativeError(NNWorking,
                                                inputs, outputs) / len(inputs)


def printCitizen(citizen):
    '''Nifty function to shorten the values in a citizen to better print on a
    single line'''
    for x in citizen[:-1]:
        print('%.3f, ' % x, end='')
    print('%.9f' % citizen[-1])


def create_net(inputs, outputs):
    '''Takes sets of inputs and outputs and maps to a hard coded NN topology'''
    global OrigAnswers
    OrigAnswers = copy.deepcopy(outputs)
    maxim = 0
    minim = 10000
    for x in outputs:
        maxim = max(maxim, max(x))
        minim = min(minim, min(x))
    EvalNN = NN.NN([0 for x in inputs[0]], [['S', 'S', 'S'], ['S', 'S']],
                   ['S' for x in outputs[0]], [0 for x in outputs[0]],
                   maxim=maxim, minim=minim)
    EvalNN.ConstructNetwork()
    return EvalNN


def mate(parents, cRate):
    '''Handler for crossover function that takes in a set of parents and cRate
    and has each parent mate with every other parent'''
    children = []
    for p1 in parents:
        for p2 in parents:
            if not p1 == p2:  # Avoid asexual reproduction
                children.append(crossover(p1, p2, cRate))
    return children


def heroFound(population, threshold):
    '''Takes a population and a test threshold and determines if a member of
    the population is actually a hero.'''
    global hero
    #print(population[0][-1], threshold)
    if population[0][-1] * 100 < threshold:
        #print('\nHero Found in Generation')
        hero = copy.deepcopy(population[0])
        # printCitizen(hero)
        return True
    return False


def train(inputs, outputs, size, participants, victors,
          generations, threshold, cRate, mRate, printFile=False):
    '''The train method takes in a set of inputs and outputs which will be
    compared against a hardcoded NN topology. The size, participants, and
    victors are with regard to tournament selection and elitism selection
    techniques. Generations is the max number of generations allowed while
    threshold is the accuracy needed. cRate and mRate are the rate of
    crossover and rate of mutation respectively. '''
    global hero
    global OrigAnswers
    EvaluationNN = create_net(inputs, outputs)
    population = generatePopulation(EvaluationNN, inputs, outputs, size)
    # Test each citizen and determine initial fitness
    evaluate(EvaluationNN, population, inputs, outputs)
    if printFile: f = open('GA.csv', 'w')
    gen = 0
    children = []
    # loop until a hero is found or we've reached max generations
    while gen <= generations and hero == 0:
        # Select our parents using tournament selection
        parents = tournament(population, participants, victors)
        # Have our parents mate (Crossover)
        children = mate(parents, cRate)
        # Have the children experience the world (Mutate)
        for child in children:
            mutate(child, mRate)
        # Test each child's fitness
        evaluate(EvaluationNN, children, inputs, outputs)
        # We were to prolific, thus children must fight to the death via draft
        # call. Make participants len(children) to have all of them fight
        # This might not be a good idea as late generation counts result in not
        # keeping the children.
        children = tournament(children, participants, victors)
        # purging of population is determined by elitism inverted on fitness
        # level (cowardace is greater number).
        # Take number of children equal to number of tournament victors and
        # reintroduce to the population
        population = sorted(population + children,
                            key=itemgetter(-1))[:-victors]
        # Determine if a child is a hero (<threshold) and if so, return child
        if heroFound(population, threshold):
            break
        else:
            print("Training: {:2.2%}".format(
                population[0][-1]), "{:2.2%}     ".format(gen / generations), end="\r")
            if printFile: f.write('%f,' % population[0][-1])
            if printFile: f.write('\n')
        gen += 1
    # return best hero if max generations is met and hero hasn't been selected.
    if printFile: f.close()
    if hero == 0:
        gen -= 1
        hero = sorted(population, key=itemgetter(-1))[0]
    EvaluationNN.SetNNWeights(hero[:-1])  # Load hero into NN, prep for usage.

    # Evaluate the hero on the inputs and outputs
    print('Generations: %d' % gen, ' ' * 20)
    print("Error Relative: {:2.5%}".format(NN.calcRelativeError(EvaluationNN, inputs, OrigAnswers)))
    print("Least Squares: %d" % NN.calcLeastSquaresError(EvaluationNN, inputs, OrigAnswers))
    for x in inputs:
        EvaluationNN.SetStartingNodesValues(x)
        EvaluationNN.CalculateNNOutputs()
        print(x, EvaluationNN.GetNNResults(), OrigAnswers[inputs.index(x)])
    print()

    return EvaluationNN


def main(inputs, outputs, size=20, participants=10, victors=5, generations=100,
         threshold=5, cRate=0.2, mRate=0.2, printFile=False):

    global OrigAnswer
    OrigAnswers = []
    global hero
    hero = 0
    eval_nn = train(inputs, outputs, size, participants, victors,
                    generations, threshold, cRate, mRate)

if __name__ == '__main__':
    print('Starting some GA training...\n')
    for i in range(1):
        #main([[2, 3], [1, 3], [3, 3]], [[101], [400], [3604]], size=20,
        #     participants=10, victors=5, generations=100000, threshold=5,
        #     cRate=0.25, mRate=0.75, printFile=False)
        main([[1,1,1,1], [1,0,1,0], [0,0,1,1]], [[2],[1],[1]], size=20, participants=10, victors=5, generations=100000, threshold=5, cRate=0.75, mRate=0.25, printFile=False)