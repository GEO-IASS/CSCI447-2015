#!/usr/local/bin/python3

import random
import operator
import sys
import math
import copy
import colors
from termcolor import colored

class ant(object):

    def __init__(self, step):
        global inputs
        global maxDimensions
        # self.location = [random.randint(0, maxDimensions) for x in
        # inputs[0]][:-1]  # Random Start
        self.location = [random.randint(
            0, maxDimensions), random.randint(0, maxDimensions)]
        self.heldValue = None  # Holds the current input block
        self.heldDensity = 10000
        self.fails = 0.0
        self.activity = random.random()
        self.stepSize = 1

    def __str__(self):
        return "ANT " + hex(id(self)) + '\t'

    def logic(self, maxDim):
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
        if self.fails / 100 < 0.99:
            self.activity += 0.01
            self.fails = 0
        else:
            self.activity -= 0.01
            self.fails = 0

    def move(self):
        global inputs
        global maxDimensions
        #print(self, 'MOVE @\t', self.location, end="  \t=> ")
        direction = random.randint(0, len(self.location) - 1)
        self.location[direction] = (
            self.location[direction] + random.choice([-1, 1])) % (maxDimensions + 1)
        #print(self.location, '%', self.heldValue)

    def pick(self, maxDim):
        global inputs
        global inputLocations
        if self.location in inputLocations and self.heldValue == None:  # Valid to pick
            # Somewhere that has a block and my hands are free
            fitness = calcFitness(self, maxDim)
            pickProb = 1 if fitness <= 1 else 1 / (fitness**2)
            #print('Pick:', pickProb, fitness)
            if random.random() < pickProb:
                self.heldValue = copy.deepcopy(
                    inputs[inputLocations.index(self.location)])
                self.heldDensity = copy.deepcopy(fitness)
                #print(self, 'PICK @\t', self.location, ' \t<=' , self.heldValue, '%', pickProb, fitness)
                del inputs[inputLocations.index(self.location)]
                inputLocations.remove(self.location)

    def drop(self, maxDim):
        global inputs
        global inputLocations
        if self.location not in inputLocations and self.heldValue != None:  # Valid to drop
            # Somewhere that doesn't have a block and I currently have a block
            fitness = calcFitness(self, maxDim, self.heldValue)
            dropProb = 1 if fitness >= 1 else fitness**4
            #print('Drop:', dropProb, fitness)
            if random.random() < dropProb:
                if self.heldDensity < fitness:
                    self.fails += 1
                #print(self, 'DROP @\t', self.location, ' \t<=', self.heldValue, '%', dropProb, fitness)
                inputs.append(copy.deepcopy(self.heldValue))
                inputLocations.append(copy.deepcopy(self.location))
                self.heldValue = None
                self.heldDensity = None

    def die(self):
        global inputs
        global maxDimensions
        global inputLocations
        bestFit = 0
        bestLoc = []
        if self.heldValue != None:
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
            #print(self, 'DROP @\t', bestLoc, ' \t<=', self.heldValue)
            inputs.append(copy.deepcopy(self.heldValue))
            inputLocations.append(copy.deepcopy(bestLoc))
            self.heldValue = None
            self.heldDensity = None


def calcDensity(location, maxDim, value=None):
    global inputs
    global inputLocations
    global maxDimensions
    diff = 0
    testing = copy.deepcopy(location)
    for i in range(-(maxDim - 4) // 2, (maxDim - 1) // 2):
        for j in range(-(maxDim - 4) // 2, (maxDim - 1) // 2):
            if i != 0 and j != 0:
                testing[0] = (location[0] + i) % (maxDimensions + 1)
                testing[1] = (location[1] + j) % (maxDimensions + 1)
                if testing in inputLocations and value == None:
                    diff += (1 / (abs(i) + (abs(j)))) * EuclideanDistance(
                        inputs[inputLocations.index(testing)], inputs[inputLocations.index(location)])
                elif testing in inputLocations and value != None:
                    diff += (1 / (abs(i) + (abs(j)))) * \
                        EuclideanDistance(
                            inputs[inputLocations.index(testing)], value)
    return diff / (((maxDim - 1) // 2)**2)


def calcFitness(thing, maxDim, value=None):
    global inputs
    global inputLocations
    global maxDimensions
    fitness = 0
    testing = copy.deepcopy(thing.location)
    for i in range(-maxDim, maxDim + 1):
        for j in range(-maxDim, maxDim + 1):
            testing[0] = int((thing.location[0] + i) % (maxDimensions + 1))
            testing[1] = int((thing.location[1] + j) % (maxDimensions + 1))
            #print(thing.location, testing)
            if testing in inputLocations and value == None:
                testFit = 1 - ((EuclideanDistance(inputs[inputLocations.index(testing)], inputs[
                               inputLocations.index(thing.location)])) / thing.activity)
                if testFit <= 0:
                    #print('Fitness:', thing, 0)
                    return 0
                else:
                    fitness += testFit
            elif testing in inputLocations and value != None:
                testFit = 1 - \
                    ((EuclideanDistance(
                        inputs[inputLocations.index(testing)], value)) / thing.activity)
                if testFit <= 0:
                    #print('Fitness:', thing, 0)
                    return 0
                else:
                    fitness += testFit
    fitness = (1 / maxDim**2) * fitness
    if fitness < 0:
        #print('Fitness:', thing, 0)
        return 0
    else:
        #print('Fitness:', thing, fitness)
        return fitness


def EuclideanDistance(val1, val2):
    #print(val1, val2)
    if len(val1) != len(val2):
        sys.exit(0)
    difference = 0
    for i in range(len(val1)):
        difference += (val1[i] - val2[i])**2
    return math.sqrt(difference)


def printColorGrid(minVal, maxVal, antSet=[]):
    global maxDimensions
    grid = 'X'
    for i in range(maxDimensions + 1):
        grid += ' ' + str(i % 10)
    grid += ' X\n'
    for x in range(maxDimensions + 1):
        grid += str(x % 10)
        for y in range(maxDimensions + 1):
            grid += ' '
            printed = 0
            for z in antSet:
                if [x, y] == z.location:
                    if printed == 0:
                        if z.heldValue != None:
                            grid += colored('+', str(colors.get_color_name(
                                rescaleArray(z.heldValue, 0, 1, minVal, maxVal))))
                            printed = 1
                        else:
                            if z.location not in inputLocations:
                                grid += '+'
                                printed = 1
            if [x, y] in inputLocations:
                if printed == 0:
                    grid += colored('X', str(colors.get_color_name(rescaleArray(
                        inputs[inputLocations.index([x, y])], 0, 1, minVal, maxVal))))
            else:
                if printed == 0:
                    grid += ' '
        grid += ' ' + str(x % 10) + '\n'
    grid += 'X'
    for i in range(maxDimensions + 1):
        grid += ' ' + str(i % 10)
    grid += ' X\n'
    return grid


def rescaleVal(val, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    return (toMinVal * (1 - ((val - fromMinVal) / (fromMaxVal - fromMinVal)))) + (toMaxVal * ((val - fromMinVal) / (fromMaxVal - fromMinVal)))


def rescaleArray(array, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    return [rescaleVal(x, fromMinVal, fromMaxVal, toMinVal, toMaxVal) for x in array]


def rescaleMatrix(matrix, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    return [rescaleArray(x, fromMinVal, fromMaxVal, toMinVal, toMaxVal) for x in matrix]


def findNear(pos):
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


def main(data, ants, iterations):
    global inputs
    # Scale the inputs
    minVal = 10000
    maxVal = 0
    for x in data:
        for y in x:
            minVal = min(minVal, y)
            maxVal = max(maxVal, y)
    inputs = rescaleMatrix(data, minVal, maxVal, 0, 1)

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
    for i in range(len(inputLocations)):
        #print(inputLocations[i], '  \t@', inputs[i])
        print(inputLocations[i], '  \t@', inputs[i], colors.get_color_name(
            rescaleArray(inputs[i], 0, 1, minVal, maxVal)))
    print()
    Original = printColorGrid(minVal, maxVal)
    print(Original)

    antCollection = []
    for a in range(ants):
        antCollection.append(
            ant(int(math.sqrt(2 * maxDimensions))))
    for i in range(iterCount):
        print('%06d ' % i, end="\r")
        for x in antCollection:
            x.logic(int(((i * 1) / iterCount) + 1))
        if i % 100 == 0:
            for x in antCollection:
                x.updateActivity()
        if i % 100 == 0:
            #print(inputs, '\n', inputLocations)
            print()
            print(printColorGrid(minVal, maxVal, antCollection))
    for x in antCollection:
        x.die()
    for i in range(len(inputLocations)):
        #print(inputLocations[i], '  \t@', inputs[i])
        print(inputLocations[i], '  \t@', colored(str(rescaleArray(inputs[i], 0, 1, minVal, maxVal)) + ' ' + str(colors.get_color_name(
            rescaleArray(inputs[i], 0, 1, minVal, maxVal))), str(colors.get_color_name(rescaleArray(inputs[i], 0, 1, minVal, maxVal)))))

    Finished = printColorGrid(minVal, maxVal)
    print()
    print('Original:')
    print(Original)
    print('Final:')
    print(Finished)

    global cluster
    cluster = []
    global clusterList
    clusterList = []

    for pos in inputLocations:
        findNear(pos)
        if clusterList[-1] != None:
            clusterList.append(None)

    # print(clusterList)

    final = []
    temp = []
    for x in clusterList:
        if x != None:
            temp.append(x)
        elif x == None:
            final.append(temp)
            temp = []

    print('Clusters:')
    for i in range(len(final)):
        print(str(i) + ': ', end="")
        for j in range(len(final[i])):
            print(data[inputLocations.index(final[i][j])], end=" ")
        print('')

if __name__ == '__main__':
    data = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [
        0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]
    main(data, 3, 10000)
