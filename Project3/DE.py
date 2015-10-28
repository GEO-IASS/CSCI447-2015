#!/usr/local/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	10/25/15
CSCI 447:	Project 3

"""

import NN
import GA
import random
import numpy
import copy
from operator import itemgetter
from operator import sub

OrigAnswers = []
hero = 0

def crossover(donorV, trialV, cRate = .5):
    offspringV = []
    indices = 0
    if(len(donorV) == len(trialV)):
        indices = len(donorV)
    forwardIndex = random.random() * indices
    offspringV.append(forwardIndex)
    for i in range(indices):
        if(i != forwardIndex):
            if (random.random() * indices) < cRate:
                offspringV.append(trialV[i])
            else:
                offspringV.append(donorV[i])
    return offspringV

def mutate(population, i, beta=.5): #several mutation schemas possible...
    trialV = []
    j = 0
    k = 0
    
    same = 0
    while(same > 2):
        j = random.random() * len(population)
        k = random.random() * len(population)
        if j != i:
            #vector2 = population[j]
            same = same + 1
        if k != i:
            #vector3 = population[k]
            same = same + 1
    trialV.append(numpy.array(population[i]) - beta * (numpy.array(population[j]) - numpy.array(population[k])))
        #(vector2[i] - vector3[i]))
    return trialV

def train(inputs, outputs, size, generations, cRate, mRate):
    global hero
    global OrigAnswers

    EvaluationNN = GA.create_net(inputs, outputs)
    #define an upper and lower bound. This is done in the NN.
    population = []
    population.append(GA.generatePopulation(EvaluationNN, inputs, outputs, size))
   # population = GA.generatePopulation(EvaluationNN, inputs, outputs, size)
    ####NEED TO FIGURE OUT THE DE VECTOR STRUCTURE SO I CAN CODE IT...
    gen = 1 #should be 0...
    trialV = []
    offspringV = []
    # loop until a hero is found or we've reached max generations
    while gen <= generations: #and hero == 0:
                
        for i in range(size):
            print('Inside of loop...') 
            # evaluate
            GA.evaluate(EvaluationNN, population, inputs, outputs)
            # mutate
            #trialV.append(mutate(population, i, mRate))
            # crossover
            #crossover(population[i], trialV, cRate)
            #selection
            #GA.evaluate(EvaluationNN, offspringV, inputs, outputs)
            #if(population[1][-1] < offspringV[-1]):
            #    population[1] = offspringV

            #if(GA.evaluate(EvaluationNN, offspringV, inputs, outputs) > GA.evaluate(EvaluationNN, population[i], inputs, outputs)):
            #    population[i] = offspringV
        gen += 1

    # return best hero if max generations is met and hero hasn't been selected.
    #hero = sorted(population, key=itemgetter(-1))[0]
    #EvaluationNN.SetNNWeights(hero[:-1])  # Load hero into NN, prep for usage.
    print()

    # Evaluate the hero on the inputs and outputs
    #for x in inputs:
    #    EvaluationNN.SetStartingNodesValues(x)
    #    EvaluationNN.CalculateNNOutputs()
    #    print(gen, x, EvaluationNN.GetNNResults(),
    #          OrigAnswers[inputs.index(x)])

    return EvaluationNN


def main(inputs, outputs, size=20, generations=100, cRate=0.5, mRate=0.5):

    eval_nn = train(inputs, outputs, size, generations, cRate, mRate)

if __name__ == '__main__':
    print('Starting some DE training...')
    # main([[2, 3]], [[101]], size=5, participants=3, victors=2,
    #     generations=500, threshold=5)
    # main([[2, 3], [1, 3]], [[101], [400]], size=9, participants=6, victors=3,
    #     generations=100000, threshold=10, cRate=0.5, mRate=0.5)
    main([[2, 3], [1, 3], [3, 3]], [[101], [400], [3604]], size=9,
         generations=1, cRate=0.5, mRate=0.5)
