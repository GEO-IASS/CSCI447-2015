#!/usr/bin/python3

import random
import numpy
import PSO

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	12/03/15
CSCI 447:	Project 4

DBSCAN algorithm implementation

Tunable parameters: 
    - eps (neighborhood size)
    - minPoints (minimum number of minPoints required to form dense region

input: inputs, eps, minPoints
output: neighborhoods
"""


def dbscan(inputs, eps, minPoints):
    '''group inputs together in neighborhoods that are close in proximity'''
    c = 0 # cluster number
    clusters = []
    for i in range(len(inputs)):
        inputs[i].append('null')
    for i in range(len(inputs)):
        if inputs[i][-1] != 'null': # if already marked, go to next input
            pass
        else:
            # mark the point as visited
            inputs[i][-1] = 'V'
            neighbors = getNeighbors(inputs, i, eps)
            # if neighbors does not have enough minPoints, mark as noise:
            if len(neighbors) < minPoints: 
                inputs[i][-1] = 'N'
            else:
#                clusters.append(neighbors)
                next_cluster = growCluster(clusters, inputs, inputs[i], neighbors, c, eps, minPoints)
                clusters.append(next_cluster)
                c+=1

    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            del clusters[i][j][-1]

    return clusters

def getNeighbors(inputs, point, eps):
    '''Find the group of neighbors around point by checking if dist < eps'''
    neighbors = []
    for i in range(len(inputs)):
        dist = PSO.EuclideanDistance(inputs[point][0:-1], inputs[i][0:-1])
        #print(dist)
        if dist < eps:
            neighbors.append(inputs[i])
    return neighbors

def growCluster(clusters, inputs, point, neighbors, c, eps, minPoints):
    '''Create the cluster based of of point's neighbors, 
       c is the current cluster in clusters'''
    inCluster = False
    cluster = []
    index = 0
#    cluster.append(point)
    # if neighbor not visited, mark as visited
    for i in range(len(neighbors)):
        index = inputs.index(neighbors[i])
        if neighbors[i][-1] != 'V':
            neighbors[i][-1] == 'V'
            n_neighbors = getNeighbors(inputs, index, eps)
            if len(n_neighbors) >= minPoints:
                n_neighbors = neighbors + n_neighbors
        for x in range(len(clusters)):
            for y in range(len(clusters[x])):
                if (neighbors[i][0:-1] == clusters[x][y][0:-1]):
                    inCluster = True
        if inCluster == False:
            cluster.append(neighbors[i])
        inCluster = False

    return cluster

def main(inputs, eps, minPoints):
    nums = []
    final_clusters = dbscan(inputs, eps, minPoints)
    print("Final Clusters: ", final_clusters)

if __name__ == '__main__':
    data = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [
        0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]
    main(data, 140, 1)
#    main([[10,10,7,8],[25,20,24,25],[1,1,1,1],[3,3,3,3],[0,0,0,0],[1,1,1,1],[3,3,3,3]], 2, 1)
