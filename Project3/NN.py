#!/usr/local/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	10/23/15
CSCI 447:	Project 3

Code for a Generic Neural Network Structure that defaults to Backpropagation.
getWeights() and setWeights() allow for this NN to work with other update functions.
For more accurate results, run on more training sets.
Input is in the format of:
([Input Vectors of Values], [Hidden Layer Arrays of Functions], Output Nodes Functions, [Output Vectors of Values], LearnRate, Threshold, Momentum
This file can be added to any project that requires a NN.
Simply import this file and call main with the specified parameters.
The returned structure is a NN that has been trained.
"""

import math
import random
from numpy import transpose
import copy


class node(object):
    '''Class for a single node in either network and in input, hidden, or output layer.
    Takes in 3 option parameters for a function and a starting value'''

    def __init__(self, appFunc='', value=0):
        self.inputs = []
        self.weights = []
        self.outputs = []
        self.error = 0
        self.func = appFunc
        self.value = value
        self.historicalWeights = []

    def addInputs(self, nodes):
        '''Add an array of input nodes to this node.
        When doing so, adds random weights for each node.'''
        for x in nodes:
            x.addOutput(self)
            self.inputs.append(x)
            self.weights.append(random.random())
            self.historicalWeights.append(0)
        self.inputs.append(node(appFunc='B', value=1))
        self.weights.append(random.random())
        self.historicalWeights.append(0)

    def addOutput(self, nodeIn):
        '''Add a node as an output to this node'''
        self.outputs.append(nodeIn)

    def setValue(self, value):
        '''Set the value of this node without calculating it'''
        self.value = value

    def setNewError(self, newError):
        '''Set the error of this node without calculating it'''
        self.error = newError

    def setWeights(self, values):
        '''Set the weights to the values provided by the list values'''
        self.weights = values

    def getInputs(self):
        '''Returns the set of input nodes for this node'''
        return self.inputs

    def getOutputs(self):
        '''Return the set of output nodes'''
        return self.outputs

    def getValue(self):
        '''Return this nodes current value'''
        return self.value

    def getError(self):
        '''Return the error of this node'''
        return self.error

    def getWeightForNode(self, nodeIn):
        '''Return the weight for a particular input node given as an input'''
        return self.weights[self.inputs.index(nodeIn)]

    def getWeightOutputs(self):
        '''Return the set of weights for the output nodes and this node'''
        temp = []
        for x in self.outputs:
            temp.append(x.getWeightForNode(self))
        return temp

    def getWeights(self):
        '''Return the set of weights for the input nodes and this node'''
        return self.weights

    def calcValue(self):
        '''Calculate the value of this node.
        This is dependent on the function this node possesses.
        summa is the summation of the values of the input nodes multiplied by their weights'''
        summa = 0
        if self.func == 'S':  # Sigmoid Function
            for x in self.inputs:
                summa += x.getValue() * self.weights[self.inputs.index(x)]
            self.value = 1 / (1 + math.exp(-summa))
        elif self.func == 'B':  # Bias Node Function
            self.value = 1
        else:
            self.value = 1

    def calcHiddenError(self):
        '''Calculate the Error assuming this is a hidden layer node summa is the summation of the errors from the output nodes multiplied by their weights.'''
        summa = 0
        if self.func == 'S':  # Sigmoid Error
            for x in self.outputs:
                summa += x.getError() * x.getWeightForNode(self)
            self.error = self.value * (1 - self.value) * summa
        else:
            self.error = 0

    def calcOutputError(self, answer):
        '''Calculate the Error assuming this is an output layer node'''
        if self.func == 'S':  # Sigmoid Error
            self.error = (answer - self.value) * self.value * (1 - self.value)
        else:
            self.error = 0

    def updateWeights(self, LearnRate, Momentum, loop):
        '''Update all of the input weights for this node via backprop.
        Requires the Learning Rate (LearnRate), Momentum, and current loop (loop) to calculate'''
        global Bloops
        #DLR = 0
        DLR = 1 - 1 / (Bloops - loop + 1)  # Linear decreasing relationship
        if self.func == 'S':
            for i in range(len(self.weights)):
                temp = self.weights[i]
                self.weights[i] += ((1 - Momentum) * max(LearnRate, DLR) * self.error * self.inputs[i].getValue()) + \
                    (Momentum * (self.weights[i] - self.historicalWeights[i]))
                self.historicalWeights[i] = temp


class NN(object):
    '''A single Neural Network that will approximate a function via an input vector, node arrangement matrix, output vector,
    answer vector, learning rate (optional) (0,1], threshold value (optional) (0,Infinity), momentum value (optional) (0,1].
    Scaling our outputs according to the domain of all our possible answers.
    New range is between 0.2 and 0.8 thus having a buffer of 0.2 on either side to accommodate an approach from that direction
    (initial guess) or the possibility of estimating an answer that exceeds the domain of our training data.'''

    def __init__(self, inputs, arrangement, outputs, answers, learnrate=0.3, threshold=1, momentum=0.5, maxim=0, minim=1000):
        self.StartingNodes = []
        self.HiddenNodes = []
        self.OutputNodes = []
        self.Threshold = 0.01 * threshold
        self.AnswerSet = answers
        self.LearnRate = learnrate
        self.converged = False
        self.Momentum = momentum
        self.inputs = inputs
        self.arrangement = arrangement
        self.outputs = outputs
        self.maxim = maxim
        self.minim = minim

    def ConstructNetwork(self):
        '''Construct the network from the inputs'''
        # Make Start Nodes
        for x in self.inputs:
            n = node(value=x)
            self.StartingNodes.append(n)
        # Make Hidden Layers
        for y in self.arrangement:
            temp = []
            for x in y:
                if self.arrangement.index(y) == 0:
                    n = node(appFunc=x)
                    n.addInputs(self.StartingNodes)
                    temp.append(n)
                else:
                    n = node(appFunc=x)
                    n.addInputs(self.HiddenNodes[self.arrangement.index(y) - 1])
                    temp.append(n)
            self.HiddenNodes.append(temp)
        # Make Output Layers
        for x in self.outputs:
            n = node(appFunc=x)
            if self.arrangement == [[]]:
                n.addInputs(self.StartingNodes)
            else:
                n.addInputs(self.HiddenNodes[-1])
            self.OutputNodes.append(n)
        # Network created and ready to function

    def SetStartingNodesValues(self, values):
        '''Reset the starting nodes values to values'''
        for i in range(len(self.StartingNodes)):
            self.StartingNodes[i].setValue(values[i])

    def SetAnswerSetValues(self, values):
        '''Reset the answerSet for the NN that is used to train against to values'''
        for i in range(len(self.AnswerSet)):
            self.AnswerSet = values
        for i in range(len(self.AnswerSet)):
            if self.maxim == self.minim:
                self.AnswerSet[i] = self.maxim / (2 * self.maxim)
            else:
                self.AnswerSet[i] = (((self.AnswerSet[i] - self.minim) * (0.8 - 0.2)) / (self.maxim - self.minim)) + 0.2

    def PrintStatus(self):
        '''Print all the values, errors, and weights contained within this NN'''
        print()
        for x in self.StartingNodes:
            print(id(x), 'has starting value:', x.getValue())
        for y in self.HiddenNodes:
            for x in y:
                print(id(x), 'has hidden value:', x.getValue())
                print(id(x), 'has hidden error:', x.getError())
                print(id(x), 'had weights:', x.getWeights())
        for x in self.OutputNodes:
            print(id(x), 'has output value:', x.getValue(), '~', self.AnswerSet[self.OutputNodes.index(x)])
            print(id(x), 'has output error:', x.getError())
            print(id(x), 'had weights:', x.getWeights())
        print()

    def CalculateNNOutputs(self):
        '''Calculate the answer of the NN'''
        for i in range(len(self.HiddenNodes)):
            for j in range(len(self.HiddenNodes[i])):
                try:
                    self.HiddenNodes[i][j].calcValue()
                except:
                    None
        for i in range(len(self.OutputNodes)):
            self.OutputNodes[i].calcValue()

    def CalculateNNErrors(self):
        '''Calculate the error from the output. Should only ever be run after ShouldBackprop() has been run.'''
        for i in range(len(list(reversed(self.HiddenNodes)))):
            for j in range(len(list(reversed(self.HiddenNodes))[i])):
                (list(reversed(self.HiddenNodes))[i][j]).calcHiddenError()

    def GetNNWeights(self):
        '''Returns the set of all the weights contained within the NN'''
        weightSet = []
        for i in range(len(self.HiddenNodes)):
            for j in range(len(self.HiddenNodes[i])):
                for x in self.HiddenNodes[i][j].getWeights():
                    weightSet.append(x)
        for i in range(len(self.OutputNodes)):
            for x in self.OutputNodes[i].getWeights():
                weightSet.append(x)
        return weightSet

    def GetNNWeightsTrim(self):
        '''Returns the set of all the weights contained within the NN after removing the bias node's weights'''
        weightSet = []
        for i in range(len(self.HiddenNodes)):
            for j in range(len(self.HiddenNodes[i])):
                for x in self.HiddenNodes[i][j].getWeights()[:-1]:
                    weightSet.append(x)
        for i in range(len(self.OutputNodes)):
            for x in self.OutputNodes[i].getWeights()[:-1]:
                weightSet.append(x)
        return weightSet

    def SetNNWeights(self, values):
        '''Sets all of the weights of the network. Mirror function for GetNNWeights'''
        counter = 0
        for i in range(len(self.HiddenNodes)):
            for j in range(len(self.HiddenNodes[i])):
                temp = []
                for k in range(len(self.HiddenNodes[i][j].getWeights())):
                    temp.append(values[counter])
                    counter += 1
                self.HiddenNodes[i][j].setWeights(temp)
        for i in range(len(self.OutputNodes)):
            temp = []
            for j in range(len(self.OutputNodes[i].getWeights())):
                temp.append(values[counter])
                counter += 1
            self.OutputNodes[i].setWeights(temp)

    def UpdateNNWeights(self, loop):
        '''Calculate the weights of the NN by forward propagation'''
        for i in range(len(self.HiddenNodes)):
            for j in range(len(self.HiddenNodes[i])):
                self.HiddenNodes[i][j].updateWeights(
                    self.LearnRate, self.Momentum, loop)
        for i in range(len(self.OutputNodes)):
            self.OutputNodes[i].updateWeights(self.LearnRate, self.Momentum, loop)

    def GetNNResults(self):
        '''Returns the values of the output nodes'''
        resultSet = []
        for i in range(len(self.OutputNodes)):
            resultSet.append((((self.OutputNodes[i].getValue() - 0.2) * (self.maxim - self.minim)) / (0.8 - 0.2)) + self.minim)
        return resultSet

    def ShouldBackprop(self):
        '''Calculates the error of the output nodes and determines if the results are within the threshold percentage.'''
        backprop = False
        for i in range(len(self.OutputNodes)):
            self.OutputNodes[i].calcOutputError(self.AnswerSet[i])
            if self.OutputNodes[i].getError()**2 > self.Threshold * 0.00000001:
                backprop = True
        self.converged = backprop
        return self.converged

    def GetNNOutputErrors(self):
        '''Returns the list of errors for the output nodes.'''
        errorSet = []
        for i in range(len(self.OutputNodes)):
            self.OutputNodes[i].calcOutputError(self.AnswerSet[i])
            errorSet.append(self.OutputNodes[i].getError()**2)
        return errorSet


def main(inputs, arrangement, outputs, answers, learnrate=0.5, threshold=1, momentum=0):
    '''Our Main Method that takes in the list of input vectors, the arrangement (topology), the list of output vectors, the list of answer vectors,
    the NN Learning Rate (learnrate), the threshold percentage (threshold), and the momentum scalar (momentum).
    Returns the NN that has been trained and is ready for testing. Testing code will be handled in the Handler File.'''
    global Bloops
    Bloops = 100000
    NNinstances = []
    OrigAnswers = copy.deepcopy(answers)

    # Max and Min of our outputs
    maxim = 0
    for x in answers:
        maxim = max(maxim, max(x))
    minim = 10000
    for x in answers:
        minim = min(minim, min(x))

    # Initial NN template that is duplicated for each input vector.
    baseNN = NN(inputs[0], arrangement, outputs, answers[0], learnrate, threshold, momentum, maxim, minim)
    baseNN.ConstructNetwork()

    # Create a copy of the template and set it's inputs and answers to the appropriate vectors.
    # Then saves this new NN as an instance in NNinstances
    for i in range(len(inputs)):
        temp = copy.deepcopy(baseNN)
        temp.SetStartingNodesValues(inputs[i])
        temp.SetAnswerSetValues(answers[i])
        NNinstances.append(temp)

    loops = 0
    while True:
        # Calculate the outputs of all instances.
        for i in range(len(NNinstances)):
            NNinstances[i].CalculateNNOutputs()
        # Calculate the output layer's error and determine if the network needs to backprop.
        done = True
        for i in range(len(NNinstances)):
            if NNinstances[i].ShouldBackprop():
                done = False
        # Merge all the weights from every instance and average the values, then reset each NN to have these new Weights
        weightSet = []
        for i in range(len(NNinstances)):
            weightSet.append(NNinstances[i].GetNNWeights())
        weightSet = transpose(weightSet)
        newWeightSet = []
        for x in weightSet:
            newWeightSet.append((sum(x)) / len(x))
        for i in range(len(NNinstances)):
            NNinstances[i].SetNNWeights(newWeightSet)
        # Calculate the output layer's error and determine if the network needs to backprop again.
        # If, at this point, neither test has failed and flipped done to false, we have a valid Weight set for use as a solution.
        for i in range(len(NNinstances)):
            if NNinstances[i].ShouldBackprop():
                done = False
        # Make sure we have iterated at least 100 times before presenting our solution. Prevents us from being lucky.
        if done and (loops >= 100):
            break
        loops += 1
        # Gets us out of this loop if we have backproped more times than our max number of loops: Bloops
        if loops > (Bloops):
            break
        # If we reach this point, then the network needs to fully backprop and update its errors and weights before recalculating its outputs.
        for i in range(len(NNinstances)):
            NNinstances[i].CalculateNNErrors()
            NNinstances[i].UpdateNNWeights(loops)

    # Select one of the finished NN's as they should all be the same and call it your final NN.
    finalNN = copy.deepcopy(NNinstances[0])

    # Test your original input vectors on the finalNN. Results should be accurate...
    for x in inputs:
        finalNN.SetStartingNodesValues(x)
        finalNN.CalculateNNOutputs()
        print(loops, x, finalNN.GetNNResults(), OrigAnswers[inputs.index(x)])
    print()

    # Ready to run tests on this NN
    return finalNN

if __name__ == '__main__':
    print('Starting some NN training...\n')

    #main([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101]], learnrate = 0.3, threshold = 5, momentum = 0.3)
    main([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400]], learnrate = 0.3, threshold = 5, momentum = 0.3)
    #for i in range(1): main([[2, 3], [1, 3], [3, 3]], [['S', 'S', 'S'], ['S', 'S']], ['S'], [[101], [400], [3604]], learnrate=0.3, threshold=5, momentum=0.3)
    #main([[1],[2],[3],[4],[5]], [['S','S','S','S','S'], ['S','S','S']], ['S'], [[1],[4],[9],[16],[25]], learnrate = 0.3, threshold = 5, momentum = 0.3)
    #main([[3],[9],[8],[2],[5],[3.9],[4.5],[1]], [['S','S','S'], ['S','S']], ['S'], [[9],[81],[64],[4],[25],[15.21],[20.25],[1]], learnrate = 0.3, threshold = 5, momentum = 0.5)
    #main([[3,4],[2,3],[4,0],[1,2],[2,4],[2,0],[2,1],[3,4]], [[]], ['S'], [[2504],[101],[25609],[100],[1],[1601],[901],[2504]], learnrate = 0.5, threshold = 5, momentum = 0.5)
