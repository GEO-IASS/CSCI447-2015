#!/usr/local/bin/python3

import random
import sys
import math
import copy


class particle(object):

    def __init__(self, dim, clusterNum):
        self.bestPosition = []
        self.bestFitness = 1000
        self.fitness = 0
        self.dimensions = dim
        self.clusters = clusterNum
        self.phiPersonal = random.random()
        self.phiGlobal = random.random()
        self.position = [random.random() for x in range(dim * clusterNum)]
        self.velocity = [random.uniform(-1, 1)
                         for x in range(dim * clusterNum)]

    def __str__(self):
        return "PARTICLE " + hex(id(self)) + ' ' + str(self.fitness) + '\t' + str(["{:1.5}".format(x) for x in self.position])

    def move(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]

    def calcVelocity(self, curIter, maxIter):
        global BestPosition
        clamp = (0.9 - 0.4) * ((maxIter - curIter) / maxIter) + 0.4
        self.phiPersonal = (
            1 / self.phiPersonal) % 1 if self.phiPersonal > 0 else 0
        self.phiGlobal = (1 / self.phiGlobal) % 1 if self.phiGlobal > 0 else 0
        for i in range(len(self.velocity)):
            GlobalUpdate = self.phiPersonal * \
                (BestPosition[i] - self.position[i])
            PersonalUpdate = self.phiGlobal * \
                (self.bestPosition[i] - self.position[i])
            self.velocity[i] = clamp * self.velocity[i] + \
                GlobalUpdate + PersonalUpdate

    def calcFitness(self, maxIter):
        global inputs
        global clusterPairs
        global BestFitness
        global BestPosition
        global Alive
        fitness = 0
        bestClusterList = []
        for i in range(len(inputs)):
            working = []
            for c in chunks(self.position, self.dimensions):
                #fitness += EuclideanDistance(inputs[i], c)
                working.append(EuclideanDistance(inputs[i], c))
            bestClusterList.append(list(chunks(self.position, self.dimensions))[
                                   working.index(min(working))])
        for i in range(len(inputs)):
            fitness += EuclideanDistance(inputs[i], bestClusterList[i])
        self.fitness = fitness
        if self.fitness < self.bestFitness:
            self.bestFitness = copy.deepcopy(self.fitness)
            self.bestPosition = copy.deepcopy(self.position)
        if self.fitness < BestFitness:
            BestFitness = copy.deepcopy(self.fitness)
            BestPosition = copy.deepcopy(self.position)
            clusterPairs = copy.deepcopy(bestClusterList)
            Alive = int(0.5 * maxIter)


def EuclideanDistance(val1, val2):
    if len(val1) != len(val2):
        print('Distance Mismatch:', val1, val2)
        sys.exit(0)
    difference = 0
    for i in range(len(val1)):
        difference += (val1[i] - val2[i])**2
    return math.sqrt(difference)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def rescaleVal(val, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    return (toMinVal * (1 - ((val - fromMinVal) / (fromMaxVal - fromMinVal)))) + (toMaxVal * ((val - fromMinVal) / (fromMaxVal - fromMinVal)))


def rescaleArray(array, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    return [rescaleVal(x, fromMinVal, fromMaxVal, toMinVal, toMaxVal) for x in array]


def rescaleMatrix(matrix, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    return [rescaleArray(x, fromMinVal, fromMaxVal, toMinVal, toMaxVal) for x in matrix]


def main(data, clusterNum, iterations):
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
    inputs = rescaleMatrix(data, minVal, maxVal, 0, 1)
    clusterPairs = [[0 for x in range(len(data[0]))] for y in range(len(data))]

    particleSet = []

    for i in range(len(inputs) * 3):
        particleSet.append(particle(len(inputs[0]), clusterNum))

    BestPosition = particleSet[0].position
    BestFitness = 10000
    Alive = 10000

    for i in range(iterations):
        #print('%2%' % i/iterations, end="\r")
        print("{:>7.2%}".format(i / iterations), end="\r")
        for p in particleSet:
            p.calcFitness(iterations)
            p.move()
            p.calcVelocity(i, iterations)
        Alive -= 1
        if Alive == 0:
            print('\nStatic system discovered...')
            break

    BestPosition = rescaleArray(BestPosition, 0, 1, minVal, maxVal)
    clusterPairs = rescaleMatrix(clusterPairs, 0, 1, minVal, maxVal)

    print('\nClusters:')
    for i in range(len(list(chunks(BestPosition, len(data[0]))))):
        print(i, list(chunks(BestPosition, len(data[0])))[i])
    print('\nResults:')
    for i in range(len(inputs)):
        print(data[i], list(chunks(BestPosition, len(inputs[0]))).index(
            clusterPairs[i]), clusterPairs[i])

    return (list(chunks(BestPosition, len(data[0]))), clusterPairs)

if __name__ == '__main__':
    data = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [
        0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]
    main(data, 3, 10000)
