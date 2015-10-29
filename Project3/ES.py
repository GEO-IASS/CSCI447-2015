#!/usr/bin/python3

import GA
import random
import numpy
import NN
import copy
from operator import itemgetter
import math

OrigAnswers = []
hero = 0

class citizen:
    reals = []
    strategies = []


class ES:
    
    def init_pop(es_net, inputs, outputs, sz):
        # ES citizens have 2 vectors: x = reals, sigma = strategies <x, sigma
        pop = []
        pop = GA.generatePopulation(es_net, inputs, outputs, sz)
        for i in range(len(pop)):
            del pop[i][-1]
            for j in range(len(pop[i])):
                pop[i].append(numpy.std(pop[i]))
            pop[i].append(0)
        return pop

    # random Gaussian N(0, 1)    
    def gauss():
#        w = 1
#        while w >= 1:
#            mu_1 = 2*random.random() - 1
#            mu_2 = 2*random.random() - 1
#            w = mu_1*mu_2 + mu_2*mu_2
#        w = math.sqrt((-2.0*numpy.log(w))/w)
        return numpy.random.normal(0, 1) #(mu_2*w)

    def mutate_vec(child):
        length = int((len(child)-1)/2)
        for i in range(length):
            child[i] = (child[i] + child[length+i]*ES.gauss())
#            x[i] = parents[i][0] if x[i] < parents[i][0]
#            x[i] = parents[i][1] if x[i] > parents[i][1]

    def mutate_strat(child): # get std devs for a child, x', pg 218
        length = int((len(child)-1)/2)
        t = math.sqrt(2.0*length)**(-1.0)
        t_prime = math.sqrt(2.0*(math.sqrt(length)))**(-1.0)
        for i in range(length, (len(child)-1)):
            child[i] = (child[i]*numpy.exp(t_prime*ES.gauss() + t*ES.gauss()))

    def mutate(child, mRate):
        ES.mutate_vec(child)
        ES.mutate_strat(child)

    def create_net(inputs, outputs):
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

def train(inputs, outputs, size, participants, victors, generations, threshold, 
                                                                 cRate, mRate):
    global hero
    global OrigAnswers
    EvaluationNN = ES.create_net(inputs, outputs)
    population = ES.init_pop(EvaluationNN, inputs, outputs, size)
    # Test each citizen and determine initial fitness
    GA.evaluate(EvaluationNN, population, inputs, outputs)
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
            ES.mutate(child, mRate)
        # Test each child's fitness
        GA.evaluate(EvaluationNN, children, inputs, outputs)
        children = GA.tournament(children, participants, victors)
        population = sorted(population + children,
                            key=itemgetter(-1))[:-victors]
        if GA.heroFound(population, threshold):
            break
        print("Training {:2.2%}".format(gen / generations), end="\r")
        gen += 1
    if hero == 0:
        hero = sorted(population, key=itemgetter(-1))[0]
    EvaluationNN.SetNNWeights(hero[:-1])  # Load hero into NN, prep for usage.
    print()

    # Evaluate the hddero on the inputs and outputs
    for x in inputs:
        EvaluationNN.SetStartingNodesValues(x)
        EvaluationNN.CalculateNNOutputs()
        print(gen, x, EvaluationNN.GetNNResults(),
              OrigAnswers[inputs.index(x)])

    return EvaluationNN

def main(inputs, outputs, size=20, members=10, victs=5, gens=100,
                                            threshold=5, cRate=0.2, mRate=0.2):
    es_nn = train(inputs, outputs, size, members, victs, gens, threshold, 
                                                                  cRate, mRate)



if __name__ == '__main__':
    print('Starting some ES training...')
    main([[2, 3], [1, 3], [3, 3]], [[101], [400], [3604]], size=20,
         members=10, victs=5, gens=100000, threshold=10, cRate=0.5, mRate=0.5)



