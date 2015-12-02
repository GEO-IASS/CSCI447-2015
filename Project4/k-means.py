#!/usr/bin/python3

import random
import numpy
import PSO

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
        if iterCount == (iterations - 1):
            pass
        else:
            centroids = updateCentroids(centroids, inputs)
        iterCount+=1

    for i in range(len(centroids)):
        del centroids[i][0]

    return centroids

def getClusters(inputs, centroids, k):
    '''Find closest cluster to each centroid via euclidean distance'''
    chosenCluster = 0
    for i in range(k):
        centroids[i][1:] = []
    for i in range(len(inputs)):
        chosenCluster = 0
        euclidean = PSO.EuclideanDistance(inputs[i], centroids[0][0])
        for j in range(k):
            tempEuc = PSO.EuclideanDistance(inputs[i], centroids[j][0])
            if tempEuc < euclidean:
                euclidean = tempEuc #PSO.EuclideanDistance(inputs[i], centroids[j][0])
                chosenCluster = j
        centroids[chosenCluster].append(inputs[i])
    return centroids

def updateCentroids(centroids, inputs):
    '''Update centroid to be geometric mean of clusters surrounding it'''
    for i in range(len(centroids)):
        if len(centroids[i]) == 1: #if centroid empty, re-initialize
            centroids[i][0] = random.choice(inputs)
        else:
            centroids[i][0] = getMean(centroids[i][1:]) #numpy.mean(centroids[i][1:])
    
    return centroids

def getMean(centroid):
    '''Get the mean of the arrays corresponding to the centroid'''
    mean = 0
    avg_centroid = []
    for i in range(len(centroid[0])):
        for j in range(len(centroid)):
            mean += (centroid[j][i]/len(centroid))
        avg_centroid.append(mean)
        mean = 0
    return avg_centroid

def main(inputs, k, iterations):
    clusters = kmeans(inputs, k, iterations)
    #print("Final Clusters: ", clusters)
    print("Final Clusters:")
    for x in clusters:
        print(x)

    return clusters

if __name__ == '__main__':
    data = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [
        0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]
    main(data, 3, 4000)
#    main([[10,10,7,8],[25,20,24,25],[1,1,1,1],[3,3,3,3],[0,0,0,0],[1,1,1,1],[3,3,3,3]], 5, 20)
