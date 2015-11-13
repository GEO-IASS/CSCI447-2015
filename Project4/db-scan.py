#!/usr/bin/python3

import random
import numpy

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
    clusters = []
    c = 0 # cluster number
    for i in range(len(inputs)):
        inputs[i].append('null')

    for i in range(len(inputs)):
        if inputs[i][-1] != 'null': # if already marked, go to next input
            pass
        else:
            # mark the point as visited
            inputs[i][-1] = 'V'
            neighbors = getNeighbors(inputs, i, eps)
            clusters.append(neighbors)
            # if neighbors does not have enough minPoints, mark as noise:
            if len(clusters[c]) < minPoints: 
                inputs[i][-1] = 'N'
            else:
                c+=1
                growCluster(inputs[i], clusters[c], c+1, eps, minPoints)
    return clusters

def getNeighbors(inputs, point, eps):
    '''Find the group of neighbors around point by checking if dist < eps'''
    neighbors = []
    p = numpy.array(inputs[point][0:-1])
    for i in inputs:
        n = numpy.array(i[0:-1])
        dist = numpy.linalg.norm(p - n)
        if dist < eps:
            neighbors.append(n)
    print(neighbors)
    return neighbors

def growCluster(point, neighbors, c, eps, minPoints):
    '''Create the cluster based of of point's neighbors, 
       c is the current cluster in clusters'''
    clusters[c].append(point)
    # if neighbor not visited, mark as visited
    for n in neighbors:
        if n[-1] != 'V':
            n[-1] == 'V'
            n_neighbors = findNeighbors(n, eps)
            if len(n_neighbors) >= minPoints:
                n_neighbors = neighbors + n_neighbors
        if n not in clusters: # this needs to be fixed?
            clusters[c].append(n)


def main(inputs, eps, minPoints):
    clusters = dbscan(inputs, eps, minPoints)
    print("Final Clusters: ", clusters)

if __name__ == '__main__':
    main([[10,10,7,8],[25,20,24,25],[1,1,1,1],[3,3,3,3],[0,0,0,0],[1,1,1,1],[3,3,3,3]], 1, 5)
