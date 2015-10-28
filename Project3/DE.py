#!/usr/local/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	10/25/15
CSCI 447:	Project 3

<<<<<<< HEAD
Code for training a Neural Network via a differential evolution 
algorithm - using the GA.py template for creating the neural 
network, generating the population, evaluating the population and 
offspring vector, checking for a hero, and evaluating the neural 
network. This code specifies the training, mutation, and crossover
necessary for differential evolution using DE/x/1/bin method.

Input:
([Vectors of Input Values], [Vectors of Output Values], population 
size, generation number, threshold, crossover rate, mutation rate)

Output:
Trained Neural Network.
=======
>>>>>>> origin/master
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

<<<<<<< HEAD
def crossover(donorV, trialV, cRate=.5):
    '''Binomial  Crossover: insures at least 1 element of the donor
    vector is carried forwared in the offspring. Otherwise, take the 
    next element from either the donor or trial vector if u(0,1) is 
    less than crossover rate.'''

=======
def crossover(donorV, trialV, cRate = .5):
>>>>>>> origin/master
    offspringV = []
    indices = 0
    if(len(donorV) == len(trialV)):
        indices = len(donorV)
<<<<<<< HEAD
    forwardIndex = random.randint(0, indices-1)
    for i in range(indices):
        if random.random() < cRate and i != forwardIndex:
            offspringV.append(trialV[i])
        else:
            offspringV.append(donorV[i])

    return offspringV

def mutate(population, i, cRate=.5): 
    '''Mutation based on population vector i. The entire population 
    is passed so two other distinct vectors can randomly be picked 
    for the difference vector.'''
    trialV = []
    j = 0
    k = 0

    # same acts as a boolean
    same = 0
    while(same < 2):
        same = 0
        j = random.randint(0, len(population)-1)
        k = random.randint(0, len(population)-1)
        if j != i:
            same = same + 1
        if k != i and k != j:
            same = same + 1

    # trialV = population[i] - cRate*(population[j] - population[k])
    trialV = list(map(lambda n: population[i][n] - cRate * (population[j][n] - population[k][n]),range(len(population[i]))))

    return trialV

def train(inputs, outputs, size, generations, threshold, cRate, mRate):
    '''The train method creates a neural netwrok from the sets of 
    inputs and outputs. A population vector of size, is initialized 
    with ranodm weight vectors associated with the weights between 
    nodes in the neural network and will be the values being trained.
    Generations is the max number of generations allowed while 
    threshold is the accuracy needed. cRate and mRate are the 
    crossover and mutation rates respectively.'''
    global hero
    global OrigAnswers

    OrigAnswers = copy.deepcopy(outputs)
    # set up NN
    EvaluationNN = GA.create_net(inputs, outputs)

    # initialize population of size as random weights of NN
    population = GA.generatePopulation(EvaluationNN, inputs, outputs, size)

    gen = 0
=======
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
>>>>>>> origin/master
    trialV = []
    offspringV = []
    # loop until a hero is found or we've reached max generations
    while gen <= generations: #and hero == 0:
<<<<<<< HEAD
        # evaluate the entire population
        GA.evaluate(EvaluationNN, population, inputs, outputs)

        for i in range(size):
            # mutate with DE/x/1/bin
            trialV = mutate(population, i, mRate)
            # perform binomial crossover 
            offspringV = crossover(population[i], trialV, cRate)
            # evaluation of offspring
            GA.evaluate(EvaluationNN, [offspringV], inputs, outputs)
            # selection of better vector
            if population[i][-1]  > offspringV[-1]:
               population[i] = offspringV

        #check for hero in population
        if GA.heroFound(population, threshold):
            break
        print("Training {:2.2%}".format(gen / generations), end="\r")
        gen += 1

    # get best vector from population
    hero = sorted(population, key=itemgetter(-1))[0]
    # Load hero into NN, prep for usage.
    EvaluationNN.SetNNWeights(hero[:-1])  
    print()

    # Evaluate the hero on the inputs and outputs
    print('Generations:', gen, 'Fitness (Sum r^2):', hero[-1])
    for x in inputs:
        EvaluationNN.SetStartingNodesValues(x)
        EvaluationNN.CalculateNNOutputs()
        print(x, EvaluationNN.GetNNResults(), OrigAnswers[inputs.index(x)])
=======
                
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
>>>>>>> origin/master

    return EvaluationNN


<<<<<<< HEAD
def main(inputs, outputs, size=20, generations=100, threshold = 10,cRate=0.5, mRate=0.5):

    eval_nn = train(inputs, outputs, size, generations, threshold, cRate, mRate)
=======
def main(inputs, outputs, size=20, generations=100, cRate=0.5, mRate=0.5):

    eval_nn = train(inputs, outputs, size, generations, cRate, mRate)
>>>>>>> origin/master

if __name__ == '__main__':
    print('Starting some DE training...')
    # main([[2, 3]], [[101]], size=5, participants=3, victors=2,
    #     generations=500, threshold=5)
    # main([[2, 3], [1, 3]], [[101], [400]], size=9, participants=6, victors=3,
    #     generations=100000, threshold=10, cRate=0.5, mRate=0.5)
<<<<<<< HEAD
    main([[2, 3], [1, 3], [3, 3]], [[101], [400], [3604]], size=20, threshold=10, generations=100000, cRate=0.3, mRate=0.6)
=======
    main([[2, 3], [1, 3], [3, 3]], [[101], [400], [3604]], size=9,
         generations=1, cRate=0.5, mRate=0.5)
>>>>>>> origin/master
