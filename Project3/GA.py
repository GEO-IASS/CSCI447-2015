#!/usr/local/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	10/23/15
CSCI 447:	Project 3
"""

import NN
import random
import copy
from operator import itemgetter


def generatePopulation(net, inputs, outputs, size):
    # Create citizens as arrays of weights that will be injected and stripped from the NN
    citizenTemp = net.GetNNWeights()
    population = []
    for i in range(size):
        for j in range(len(citizenTemp)):
            citizenTemp[j] = random.random()  # Random weights for the NN topology
        population.append(copy.deepcopy(citizenTemp))
        population[-1].append(0)  # Each citizen tracks their current heroic level based on the dimensionality of the outputs
    # for i in population:
    #    print(i)
    # print()
    # for i in population:
    #    print(i[:-1])
    # print()
    return population


def crossover(parent1, parent2, rate=0.2):
    child = []
    current = 0
    # for i in range(len(parent1[:-1])):
    for i in range(len(parent1[:-2])):
        if random.random() < rate:
            current = 1
        else:
            current = 0
        if current == 0:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    child.append(0)  # child has not yet tested
    return child


def mutate(child, rate=0.2):
    # for i in range(len(child[:-1])):
    for i in range(len(child[:-2])):
        if random.random() < rate:  # chance child will experience something that enlightens them
            child[i] += (random.random() * 1) - 0.5


def tournament(population, participants, victors):
    # Treat this as a minimization problem with regards to error.
    bracket = sorted(random.sample(population, participants), key=itemgetter(-1))
    return bracket[0:victors]


def evaluate(NN, group, inputs, outputs):
    for citizen in group:
        citizen[-1] = 0
        NN.SetNNWeights(citizen[:-1])
        ans = []
        for i in range(len(inputs)):
            NN.SetStartingNodesValues(inputs[i])
            NN.SetAnswerSetValues(copy.deepcopy(outputs[inputs.index(inputs[i])]))
            NN.CalculateNNOutputs()
            citizen[-1] += sum(list(map(lambda x: abs(NN.GetNNResults()[x] - outputs[inputs.index(inputs[i])][x]), range(len(NN.GetNNResults())))))
            #print(NN.GetNNResults()[0], outputs[i][0], NN.GetNNResults()[0]-outputs[i][0])
            ans.append(NN.GetNNResults() + outputs[i] + list(map(lambda x: abs(NN.GetNNResults()[x] - outputs[i][x]), range(len(NN.GetNNResults())))))
        citizen.append(ans)


def printCitizen(citizen):
    # for x in citizen[:-1]:
    for x in citizen[:-2]:
        print('%.3f, ' % x, end='')
    #print('%.9f' % citizen[-1])
    print('%.9f, ' % citizen[-2])
    print(citizen[-1])


def main(inputs, outputs, size=20, participants=10, victors=5, generations=10, threshold=5, cRate=0.2, mRate=0.2):

    OrigAnswers = copy.deepcopy(outputs)
    # Make base NN for template and fitness evaluations
    # Max and Min of our outputs
    maxim = 0
    for x in outputs:
        maxim = max(maxim, max(x))
    minim = 10000
    for x in outputs:
        minim = min(minim, min(x))
    EvaluationNN = NN.NN([0 for x in inputs[0]], [['S', 'S', 'S'], ['S', 'S']], ['S' for x in outputs[0]],
                         [0 for x in outputs[0]], threshold=threshold, maxim=maxim, minim=minim)
    EvaluationNN.ConstructNetwork()
    # EvaluationNN.PrintStatus()
    population = generatePopulation(EvaluationNN, inputs, outputs, size)

    # Test each citizen and determine heroic level (make this a function?)
    print(inputs, outputs)
    evaluate(EvaluationNN, population, inputs, outputs)

    gen = 0
    hero = 0
    children = []
    # loop until a hero is found or we've reached max generations
    # Keep track of generation number starting at 0 and ended at our max generation count
    while gen <= generations and hero == 0:
        print('\n\n\nGeneration:', gen)
        print('Old Population:')
        for citizen in population:
            printCitizen(citizen)
        # Citizens are selected via random distribution and ranked based on heroics where we'll pull some number of potential parents
        parents = tournament(population, participants, victors)
        # Have our parents mate (Crossover)
        # 2 parents where a random number of crossovers occur at random locations based on a passed in paramenter likelyhood
        children = []
        for p1 in parents:
            for p2 in parents:
                children.append(crossover(p1, p2, cRate))
        # Have the children experience the world (Mutate)
            # random number of mutations that occur on random weights based on a passed in parameter of likelyhood
        for child in children:
            mutate(child, mRate)
        # Test each child's heroic level
        evaluate(EvaluationNN, children, inputs, outputs)
        # We were to prolific, thus children must fight to the death via draft call. Make participants len(children) to have all of them fight
        children = tournament(children, participants, victors)
        print('\nChildren')
        for child in children:
            printCitizen(child)
        # purging of population is determined by elitism inverted on heroic level (cowardace is greater number).
        # Take number of children equal to number of tournament victors and reintroduce to the population
        population = sorted(population + children, key=itemgetter(-2))[:-victors]
        print('\nNew Population:')
        for citizen in population:
            printCitizen(citizen)

        # Determine if a child is a hero (<threshold) and if so, return child (break)
        # if population[0][-1] < threshold * 0.00000001:  # Check if answer is acceptable
        if population[0][-2] < threshold * 0.01:
            print('\nHero Found in Generation', gen)
            hero = copy.deepcopy(population[0])
            printCitizen(hero)
            break
        # Increment Generation counter
        gen += 1
    # return best hero we have if max generations is met.
    # hero = sorted(population, key=itemgetter(-1))[0]  # default to best in population if no hero steps forward
    hero = sorted(population, key=itemgetter(-2))[0]
    # EvaluationNN.SetNNWeights(hero[:-1])
    EvaluationNN.SetNNWeights(hero[:-2])
    print()

    for x in inputs:
        EvaluationNN.SetStartingNodesValues(x)
        EvaluationNN.CalculateNNOutputs()
        EvaluationNN.SetAnswerSetValues(copy.deepcopy(OrigAnswers[inputs.index(x)]))
        print(x, EvaluationNN.GetNNResults(), OrigAnswers[inputs.index(x)])
        EvaluationNN.PrintStatus()
    print()

    # Evaluate the hero on the inputs and outputs

if __name__ == '__main__':
    print('Starting some GA training...')

    #main([[2, 3]], [[101]], size=5, participants=3, victors=2, generations=500, threshold=5)
    main([[2, 3], [1, 3]], [[101], [400]], size=9, participants=6, victors=3, generations=10000, threshold=5, cRate=0.5, mRate=0.5)
    #main([[2, 3], [1, 3], [3, 3]], [[101], [400], [3604]], size=9, participants=6, victors=3, generations=100)
