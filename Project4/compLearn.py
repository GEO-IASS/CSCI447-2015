#!/usr/bin/python3

import random
import numpy
import math

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	12/03/15
CSCI 447:	Project 4

Competitive Learning Neural Network algorithm implementation

Tunable parameters: 
    - numHNodes: number of hidden nodes
    - # iterations

input: dataset, numHNodes
output: winner
- normalize all inputs
- Send inputs to each hidden node, apply random weights between 0 and 1
- nodeOutput = sum(weight[i]*input[i])
- highest output node is the winner (winner takes all): i = argmax[....]
- update winner: w[i] = w[i] + nx^(n 
- normalize winner: w[i] = w[i]/||w[i]||
- repeat for a number of iterations

tunable parameters:
    numHNodes: number of hidden nodes in hidden layer
"""

def sigmoid(node):
   return 1 / (1 + math.exp(-node))

def competitiveLearn(inputs, numHNodes, iterations, learnRate):
    index = 0
    nodes= []
    winner = []
    weights = []
    wcount = 0
    temp_weights = []
    tmp_wt = []
##    for x in inputs:
##        x = numpy.linalg.norm(x) # normalize inputs
##        print("Normalized input vector: ", x)
#        weights.append(random.random())
#        weights[wcount] = numpy.linalg.norm(weights[wcount])
#        wcount+=1
    # randomly assign numHNodes input vectors to be the weights
    for i in range(numHNodes):
        weights.append(random.choice(inputs))
        weights[i] = numpy.linalg.norm(weights[i])
    temp_weights = weights
##        weights[i] = numpy.linalg.norm(weights[i])

    for i in range(iterations):
        # randomly select an input vector for comparison
        selectedInput = random.choice(inputs)
        # normalize the chosen input
        selectedInput = numpy.linalg.norm(selectedInput)
        # calculate a starting point for the winner
        winner = weights[0] * selectedInput
        # find the winning weight vector
        for w in weights:
            index = weights.index(w)
            tmp_w = w
            temp = tmp_w*selectedInput
            if winner >= temp:
                winner = temp
        weights[index] = learnRate*(math.fabs(selectedInput - weights[index]))
    temp_inputs = []
    tmp_i = []
    # normalize inputs to compare against weights
    for i in inputs:
        tmp_i = i
        temp_inputs.append(numpy.linalg.norm(tmp_i))
    # weights are now the cluster centers, TODO: calc distances for each input
    return weights, temp_inputs


# numHNodes is number hidden nodes aka length of weight vector
def main(inputs, numHNodes, iterations, learnRate):
    clusters = []
    clusters = competitiveLearn(inputs, numHNodes, iterations, learnRate)
    #TODO: calculate distance to get which cluster center the inputs lie in
    print("Cluster centers (normalized): ", clusters[0], 
                                        "\nValues (normalized): ", clusters[1])

if __name__ == '__main__':
#    main([[10,10,7,8],[25,20,24,25],[1,1,1,1],[3,3,3,3],[2,2,2,2],[1,1,1,1],
#         [3,3,3,3]], 2, 500, 0.2)
    main([[300,200],[20,10],[50,70],[5,1]], 4, 5000, 0.2)
