#!/usr/bin/python3

import random
import numpy
import math
import PSO

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
# not sure if this is needed - LT
def sigmoid(node):
   return 1 / (1 + math.exp(-node))

def competitiveLearn(inputs, numHNodes, iterations, learnRate):
    index = 0
    nodes= []
    winner = []
    weights = []
    wcount = 0
    tmp_wt = []
    minWt = 10000
    maxWt = 0
    # randomly assign numHNodes input vectors to be the weights
    for i in range(numHNodes):
        weights.append(random.choice(inputs))
    for i in range(iterations):
        # randomly select an input vector for comparison
        selectedInput = random.choice(inputs)
        # calculate a starting point for the winner
        winner = PSO.EuclideanDistance(weights[0], selectedInput)
        # find the winning weight vector
        index = 0
        for w in weights:
            tmp_w = w
            temp = PSO.EuclideanDistance(tmp_w, selectedInput)
            if winner >= temp: # want the shortest distance
                winner = temp
                index = weights.index(w) # winning index
        # update the weight at the winning index
        dist = learnRate*(PSO.EuclideanDistance(selectedInput, weights[index]))
        for j in range(len(weights[index])):
            weights[index][j] += learnRate*(dist)
        for x in weights:
            for y in x:
                minWt = min(minWt, y)
                maxWt = max(maxWt, y)
        weights = PSO.rescaleMatrix(weights, minWt, maxWt, 0, 1)
    # weights are now the cluster centers
    return weights


# numHNodes is number hidden nodes aka length of weight vector
def main(inputs, numHNodes, iterations, learnRate):
    clusters = []
    cluster_num = 0
    final_clusters = []
    inputs_copy = []
    # normalize inputs
    minIn = 10000
    maxIn = 0
    for x in inputs:
        for y in x:
            minIn = min(minIn, y)
            maxIn = max(maxIn, y)
    print("Max: ", maxIn, ", Min: ", minIn)
    inputs_copy = PSO.rescaleMatrix(inputs, minIn, maxIn, 0, 1)
    clusters = competitiveLearn(inputs_copy, numHNodes, iterations, learnRate)
    print("Clusters: ", clusters)
    for c in range(len(clusters)):
        final_clusters.append([])
    # calculate distance to get which cluster center the inputs lie in
    dist = 10000
    for i in range(len(inputs_copy)):
        for j in range(len(clusters)):
            tmpDist = PSO.EuclideanDistance(clusters[j], inputs_copy[i])
            if tmpDist < dist:
                dist = tmpDist
                cluster_num = j
        dist = 10000
        final_clusters[cluster_num].append(inputs_copy[i])
    print("Clusters: ")
    for i in range(len(final_clusters)):
        print(final_clusters[i])
        

if __name__ == '__main__':
    data = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [
        0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]
    main(data, 5, 200, 0.95)

#    main([[10,10,7,8],[25,20,24,25],[1,1,1,1],[3,3,3,3],[0,0,0,0],[1,1,1,1],[3,3,3,3]], 5, 200, 0.05)
