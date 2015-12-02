#!/usr/local/bin/python3

import random
import sys
import math
import copy

"""
Author:     Clint Cooper, Emily Rohrbough, Leah Thompson
Date:       12/03/15
CSCI 447:   Project 4

Particle Swarm Optimization Implementation

Tunable parameters: 
    - # of swarms partilces should be expecting (Upper bound)
    - # iterations

input:  N examples of real-vector observations (x1, ..., xn), # clusters, # iterations
output: clusters of N real-vector observations
"""

class particle(object):
    '''A particle class that holds a position in space as it explores the space with
    it's siblings. Keeps track of some number of particles that represent the best
    cluster thus far.'''

    def __init__(self, dim, clusterNum):
        '''Start the particle in a random location in the space with a little starting
        speed in a random direction.'''
        self.bestPosition = []
        self.bestFitness = 1000
        self.fitness = 0
        self.dimensions = dim
        self.clusters = clusterNum
        self.phiPersonal = random.random()
        self.phiGlobal = random.random()
        self.position = [random.random() for x in range(dim * clusterNum)] # Start Location
        self.velocity = [random.uniform(-1, 1) for x in range(dim * clusterNum)] # Start speed

    def __str__(self):
        return "PARTICLE " + hex(id(self)) + ' ' + str(self.fitness) + '\t' + str(["{:1.5}".format(x) for x in self.position])

    def move(self):
        '''Moves particle by velocity values in space'''
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]

    def calcVelocity(self, curIter, maxIter):
        '''Updates the velocity of this particle. Takes in how many iterations
        have passed and the total number of iterations. Uses a clamping value
        based on how long it's been alive. Also updates phiPersonal and
        phiGlobal using a chaos map to give it some "jiggle". '''
        global BestPosition
        clamp = (0.9 - 0.4) * ((maxIter - curIter) / maxIter) + 0.4 # Clamp value from papar
        # Update phi's using Guass Chaos Map
        self.phiPersonal = (1 / self.phiPersonal) % 1 if self.phiPersonal > 0 else 0
        self.phiGlobal = (1 / self.phiGlobal) % 1 if self.phiGlobal > 0 else 0
        for i in range(len(self.velocity)):
            # Bring it all together for each dimension
            GlobalUpdate = self.phiPersonal * (BestPosition[i] - self.position[i])
            PersonalUpdate = self.phiGlobal * (self.bestPosition[i] - self.position[i])
            self.velocity[i] = clamp * self.velocity[i] + GlobalUpdate + PersonalUpdate

    def calcFitness(self, maxIter):
        '''Determines the particles current fitness in the space. This uses
        euclidean distance as the distance calculation. Also, if the new 
        fitness is better than the personal best, update. Also, if the new
        fitness is better than the worst of the global set, replace. Alive
        is a representation of how long the system is allowed to be static.'''
        global inputs
        global clusterPairs
        global BestFitness
        global BestPosition
        global Alive
        fitness = 0
        bestClusterList = []
        for i in range(len(inputs)):
            working = []
            # Propose clusters
            for c in chunks(self.position, self.dimensions):
                working.append(EuclideanDistance(inputs[i], c))
            # Add closest cluster to bestCluserList for this input
            bestClusterList.append(list(chunks(self.position, self.dimensions))[working.index(min(working))])
        for i in range(len(inputs)):
            # Calc overall fitness for each input and it's choosen cluster
            fitness += EuclideanDistance(inputs[i], bestClusterList[i])
        self.fitness = fitness
        # Update personal fitness and positioning
        if self.fitness < self.bestFitness:
            self.bestFitness = copy.deepcopy(self.fitness)
            self.bestPosition = copy.deepcopy(self.position)
        # Update global fitness, positioning and reset system TTL
        if self.fitness < BestFitness:
            BestFitness = copy.deepcopy(self.fitness)
            BestPosition = copy.deepcopy(self.position)
            clusterPairs = copy.deepcopy(bestClusterList)
            Alive = int(0.5 * maxIter)


def EuclideanDistance(val1, val2):
    '''Calculate euclidean distance between val1 and val2 for n dimensions.'''
    if len(val1) != len(val2):
        print('Distance Mismatch:', val1, val2)
        sys.exit(0)
    difference = 0
    for i in range(len(val1)):
        difference += (val1[i] - val2[i])**2
    return math.sqrt(difference)


def chunks(l, n):
    '''Returns the clusters from a flattened array.'''
    for i in range(0, len(l), n):
        yield l[i:i + n]


def rescaleVal(val, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    '''Rescales a single value using each domain min and max values.'''
    return (toMinVal * (1 - ((val - fromMinVal) / (fromMaxVal - fromMinVal)))) + (toMaxVal * ((val - fromMinVal) / (fromMaxVal - fromMinVal)))


def rescaleArray(array, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    '''Rescales an array using each domain min and max values.'''
    return [rescaleVal(x, fromMinVal, fromMaxVal, toMinVal, toMaxVal) for x in array]


def rescaleMatrix(matrix, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    '''Rescales a matrix using each domain min and max values.'''
    return [rescaleArray(x, fromMinVal, fromMaxVal, toMinVal, toMaxVal) for x in matrix]


def PSO(data, clusterNum, iterations):
    '''Takes some input data matrix and a max number of clusters, plus iterations.
    Generates particles and lets them move through the solution space. 
    Once they have stopped improving or TTL has expired, we return the set of
    clusters and their pairings with the input data.'''
    global inputs
    global clusterPairs
    global BestPosition
    global BestFitness
    global Alive
    # Scale the inputs
    minVal = 10000
    maxVal = 0
    for x in data:
        for y in x:
            minVal = min(minVal, y)
            maxVal = max(maxVal, y)
    inputs = rescaleMatrix(data, minVal, maxVal, 0, 1) # Scale input data
    # Initialize some clusters
    clusterPairs = [[0 for x in range(len(data[0]))] for y in range(len(data))]

    # Initialize some particles
    particleSet = []
    for i in range(len(inputs) * 3):
        particleSet.append(particle(len(inputs[0]), clusterNum))

    # Initialize (guess)
    BestPosition = particleSet[0].position
    BestFitness = 10000
    Alive = 10000

    # Run until we're static for a long time or we finish.
    for i in range(iterations):
        #print('%2%' % i/iterations, end="\r")
        #print("{:>7.2%}".format(i / iterations), end="\r")
        for p in particleSet:
            p.calcFitness(iterations)
            p.move()
            p.calcVelocity(i, iterations)
        Alive -= 1
        if Alive == 0:
            print('\nStatic system discovered...')
            break

    # Return tuple of clusters and their matches
    BestPosition = rescaleArray(BestPosition, 0, 1, minVal, maxVal)
    clusterPairs = rescaleMatrix(clusterPairs, 0, 1, minVal, maxVal)

    finalClusters = [[] for x in range(clusterNum)]
    for i in range(len(data)):
        finalClusters[(list(chunks(BestPosition, len(data[0])))).index(clusterPairs[i])].append(data[i])

    return finalClusters

def main(data, clusterNum, iterations):
    clusters = PSO(data, clusterNum, iterations)
    print('Clusters:')
    for x in clusters:
        print(x)

if __name__ == '__main__':
    data = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [
        0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]
    main(data, 3, 10000)
