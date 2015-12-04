#!/usr/local/bin/python3

import random
import sys
import math
import copy
import PSO

"""
Author:     Clint Cooper, Emily Rohrbough, Leah Thompson
Date:       12/03/15
CSCI 447:   Project 4

Ant Colony Optimization Implementation

Tunable parameters: 
    - # of ants 
    - # iterations

input:  N examples of real-vector observations (x1, ..., xn), # ants, # iterations
output: clusters of N real-vector observations
"""


class ant(object):
    '''An ant class that holds a position in a 2D space as it explores this space with
    it's associates. Has psuedo short term memory to track at what fitness the current
    held value was located, so we can assess if we were better or worse...'''

    def __init__(self, step):
        '''Start the ant at a random location on the space and allow movement by stepsize.'''
        global inputs
        global maxDimensions
        # Random start
        self.location = [random.randint(0, maxDimensions), random.randint(0, maxDimensions)]
        self.heldValue = None  # Holds the current input block
        self.heldDensity = 10000
        self.bads = 0.0
        self.totals = 0.0
        self.activity = random.random()  # Random eagerness value
        self.stepSize = 1

    def __str__(self):
        return "ANT " + hex(id(self)) + '\t'

    def logic(self, maxDim):
        '''Try to pick a value if it has none, if it does have one, try to drop it. If
        it didn't pickup or drop, move by stepSize.'''
        if self.heldValue == None:
            self.pick(maxDim)
            if self.heldValue == None:
                for i in range(self.stepSize):
                    self.move()
        elif self.heldValue != None:
            self.drop(maxDim)
            if self.heldValue != None:
                for i in range(self.stepSize):
                    self.move()
        else:
            print('Whoops')

    def updateActivity(self):
        '''Updates activity value based on how correct it has been so far.'''
        if self.totals > 0:
            if self.bads / self.totals > 0.75:
                self.activity += 0.01
                self.bads = 0
                self.totals = 0
            else:
                self.activity -= 0.01
                self.bads = 0
                self.totals = 0

    def move(self):
        '''Move the ant in within the restricted space in 2D space in a random direction.'''
        global inputs
        global maxDimensions
        #print(self, 'MOVE @\t', self.location, end="  \t=> ")
        direction = random.randint(0, len(self.location) - 1)
        self.location[direction] = (self.location[direction] + random.choice([-1, 1])) % (maxDimensions + 1)
        #print(self.location, '%', self.heldValue)

    def pick(self, maxDim):
        '''Look at the current location and determine how well the object currently fits.
        If it doesn't fit very well, probabilistically pick it up and carry it.'''
        global inputs
        global inputLocations
        if self.location in inputLocations and self.heldValue == None:  # Valid to pick
            # Somewhere that has a block and my hands are free
            fitness = calcFitness(self, maxDim)
            pickProb = 1 if fitness <= 1 else 1 / (fitness**2)
            #print('Pick:', pickProb, fitness)
            if random.random() < pickProb:
                self.heldValue = copy.deepcopy(inputs[inputLocations.index(self.location)])
                self.heldDensity = copy.deepcopy(fitness)
                #print(self, 'PICK @\t', self.location, ' \t<=' , self.heldValue, '%', pickProb, fitness)
                del inputs[inputLocations.index(self.location)]
                inputLocations.remove(self.location)

    def drop(self, maxDim):
        '''Look at the current location and determine how well the object would fit here.
        If it fits relatively well, probabilistically drop it at this location.'''
        global inputs
        global inputLocations
        if self.location not in inputLocations and self.heldValue != None:  # Valid to drop
            # Somewhere that doesn't have a block and I currently have a block
            fitness = calcFitness(self, maxDim, self.heldValue)
            dropProb = 1 if fitness >= 1 else fitness**4
            #print('Drop:', dropProb, fitness)
            if random.random() < dropProb:
                self.totals += 1
                if self.heldDensity < fitness:
                    self.bads += 1
                #print(self, 'DROP @\t', self.location, ' \t<=', self.heldValue, '%', dropProb, fitness)
                inputs.append(copy.deepcopy(self.heldValue))
                inputLocations.append(copy.deepcopy(self.location))
                self.heldValue = None
                self.heldDensity = None

    def die(self):
        '''Once the ant has reached the end of it's life, we give it a chance to be enlightened.
        The ant will explore all locations on the grid systematical and determine the best location
        for the currently held block. Once this is determined, it is placed there.'''
        global inputs
        global maxDimensions
        global inputLocations
        bestFit = 0
        bestLoc = []
        if self.heldValue != None:
            # We have a location that is obvously better
            for i in range(maxDimensions + 1):
                for j in range(maxDimensions + 1):
                    testLoc = [i, j]
                    if testLoc not in inputLocations:
                        self.location = testLoc
                        fitness = calcFitness(self, 1, self.heldValue)
                        #print(bestFit, fitness)
                        if fitness > bestFit:
                            bestFit = fitness
                            bestLoc = [i, j]
            # We need to look for the best place now that an obvious one doesn't exist
            if bestFit == 0:
                bestFit = 1000
                for i in range(maxDimensions + 1):
                    for j in range(maxDimensions + 1):
                        testLoc = [i, j]
                        if testLoc not in inputLocations:
                            diff = 0
                            for k in range(-5, 6):
                                for l in range(-5, 6):
                                    testPos = copy.deepcopy(testLoc)
                                    testPos[0] = int(
                                        (testLoc[0] + k) % (maxDimensions + 1))
                                    testPos[1] = int(
                                        (testLoc[1] + l) % (maxDimensions + 1))
                                    if testPos in inputLocations:
                                        diff += PSO.EuclideanDistance(
                                            self.heldValue, inputs[inputLocations.index(testPos)])
                            if diff < bestFit:
                                bestFit = diff
                                bestLoc = [i, j]
            #print(self, 'DROP @\t', bestLoc, ' \t<=', self.heldValue)
            inputs.append(copy.deepcopy(self.heldValue))
            inputLocations.append(copy.deepcopy(bestLoc))
            self.heldValue = None
            self.heldDensity = None


def calcFitness(thing, maxDim, value=None):
    '''Calculate the fitness of a thing based on it's current location, the view size, and
    the other things that exist within that range. Use EuclideanDistance to determine intra-thing
    similarity. Several conditions exist where we want to punish so fitness is bottomed to 0.'''
    global inputs
    global inputLocations
    global maxDimensions
    fitness = 0
    testing = copy.deepcopy(thing.location)
    for i in range(-maxDim, maxDim + 1):
        for j in range(-maxDim, maxDim + 1):
            testing[0] = int((thing.location[0] + i) % (maxDimensions + 1))
            testing[1] = int((thing.location[1] + j) % (maxDimensions + 1))
            if testing in inputLocations and value == None:
                testFit = 1 - ((PSO.EuclideanDistance(inputs[inputLocations.index(testing)], inputs[
                               inputLocations.index(thing.location)])) / thing.activity)
                if testFit <= 0: # If we ever go negative, bail
                    return 0
                else:
                    fitness += testFit
            elif testing in inputLocations and value != None:
                testFit = 1 - \
                    ((PSO.EuclideanDistance(
                        inputs[inputLocations.index(testing)], value)) / thing.activity)
                if testFit <= 0: # If we ever go negative, bail
                    return 0
                else:
                    fitness += testFit
    #fitness = (1 / 2) * fitness
    if fitness <= 0: # If the final sum is negative
        return 0
    else:
        return fitness


def findNear(pos):
    '''Used to find clusters from the finished set. This uses an adjacency exploration technique
    to determine the "islands" or clusters that exist within the 2D space.'''
    global cluster
    global clusterList
    global maxDimensions
    toCheck = [pos]
    while len(toCheck) > 0:
        value = toCheck.pop()
        for i in range(-1, 2):
            for j in range(-1, 2):
                testPos = copy.deepcopy(value)
                testPos[0] = int((value[0] + i) % (maxDimensions + 1))
                testPos[1] = int((value[1] + j) % (maxDimensions + 1))
                if testPos in inputLocations and testPos not in cluster and testPos not in clusterList:
                    cluster.append(testPos)
                    toCheck.append(testPos)
        # print(toCheck)
    for x in cluster:
        clusterList.append(x)
    cluster = []


def ACO(data, ants, iterations):
    '''Take some input matrix of initial data points and values for the number of ants and
    the number of iterations. Data is randomly placed on a 2D field that is directly related
    to the number of input values. This keeps a desirable density on the field at all times.
    Ants carry out their actions, one per iteration, and then die. The points are then assessed
    and clusters are determiend.'''
    global inputs
    # Scale the inputs
    minVal = 10000
    maxVal = 0
    for x in data:
        for y in x:
            minVal = min(minVal, y)
            maxVal = max(maxVal, y)
    inputs = PSO.rescaleMatrix(data, minVal, maxVal, 0, 1)

    iterCount = len(data) * iterations

    global maxDimensions
    maxDimensions = int(math.sqrt(10 * len(inputs)) + 0.5)
    global inputLocations
    inputLocations = []
    # Every input needs a location
    for x in inputs:
        pos = [random.randint(0, maxDimensions),
               random.randint(0, maxDimensions)]
        while pos in inputLocations:
            pos = [random.randint(0, maxDimensions),
                   random.randint(0, maxDimensions)]
        inputLocations.append(pos)
    # for i in range(len(inputLocations)):
    #    print(inputLocations[i], '  \t@', inputs[i])

    antCollection = []
    for a in range(ants):
        antCollection.append(ant(int(math.sqrt(2 * maxDimensions))))
    for i in range(iterCount):
        print("{:>7.2%}".format(i / iterCount), end="\r")
        for x in antCollection:
            x.logic(int(((i * 5) / iterCount) + 1))
        if i % 100 == 0:
            for x in antCollection:
                x.updateActivity()
    for x in antCollection:
        x.die()
    # for i in range(len(inputLocations)):
    #    print(inputLocations[i], '  \t@', inputs[i])

    global cluster
    cluster = []
    global clusterList
    clusterList = []

    #print(inputLocations, len(inputLocations))
    for pos in inputLocations:
        findNear(pos)
        if clusterList[-1] != None:
            clusterList.append(None)

    # print(clusterList)

    finalClusters = []
    temp = []
    for x in clusterList:
        if x != None:
            temp.append(data[inputLocations.index(x)])
        elif x == None:
            finalClusters.append(temp)
            temp = []

    return finalClusters


def main(data, ants, iterations):
    clusters = ACO(data, ants, iterations)
    print('Clusters:')
    for x in clusters:
        print(x)

if __name__ == '__main__':
    data = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [
        0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]
    main(data, 2, 100000)
