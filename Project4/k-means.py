#!/usr/bin/python3

import random
import numpy

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	12/03/15
CSCI 447:	Project 4

k-means algorithm implementation

Tunable parameters: 
    - k (must be between 1 and N - 1), k cluster and k starting points
    - # iterations

input: N examples of real-vector observations (x1, ..., xn), K partitions
output: updated means
"""

def kmeans(inputs, k, iterations):
    ''' Make initial guesses of means: M = {m1,...,mk}
    while means are changing:
        classify samples into clusters based on guessed means
        for i=1 to k:
            replace mi with mean of all samples for cluster i'''
    centroids = []
    clusters = []
    temp = []
    iterCount = 0
    #select random start points from inputs
    for i in range(k):
        temp.append(random.choice(inputs))
        centroids.append(temp)
        temp = []

    while iterCount < iterations:
        #find closest cluster to each points via euclidean distance
        clusters = getClusters(inputs, centroids, k)
        #update centroids based on new clusters
        centroids = updateCentroids(centroids, inputs)
        iterCount+=1

    return centroids

def getClusters(inputs, centroids, k):
    '''Find closes cluster to each centroid via euclidean distance'''
    chosenCluster = 0
    for i in range(len(inputs)-1):
        a = numpy.array(inputs[i])
        euclidean = numpy.linalg.norm(a - centroids[0][0])
        for j in range(k):
            centroids[i][1:-1] = []
            b = numpy.array(centroids[j][0])
            tempEuc = numpy.linalg.norm(a - b)
            if tempEuc < euclidean:
                euclidean = numpy.linalg.norm(a - b)
                chosenCluster = j
        centroids[chosenCluster].append(a)
    return centroids

def updateCentroids(centroids, inputs):
    '''Update centroid to be geometric mean of clusters surrounding it'''
    for i in range(len(centroids)):
        if len(centroids[i]) == 1: #if centroid empty, re-initialize
            centroids[i][0] = random.choice(inputs)
        else:
            centroids[i][0] = numpy.mean(centroids[i][1])
    
    return centroids
        

def main(inputs, k, iterations):
    clusters = kmeans(inputs, k, iterations)
    print("Final Clusters: ", clusters)

if __name__ == '__main__':
    main([[1,1,1,1],[1,0,1,0],[0,0,1,1]], 2, 20)
