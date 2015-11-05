#!/usr/local/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	11/01/15
CSCI 447:	Project 3

Code for a Generic Neural Network Structure that defaults to Backpropagation.
getWeights() and setWeights() allow for this NN to work with other update
functions.
For more accurate results, run on more training sets.
Input is in the format of:
([Input Vectors of Values], [Hidden Layer Arrays of Functions], Output Nodes
Functions, [Output Vectors of Values], LearnRate, Threshold, Momentum)
This file can be added to any project that requires a NN.
Simply import this file and call main with the specified parameters.
The returned structure is a NN that has been trained.
"""

import math
import random
from numpy import transpose
from scipy.special import expit
import copy


class node(object):
    '''Class for a single node in either network and in input, hidden, or
    output layer. Takes in 3 option parameters for a function and a starting
    value.'''

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
        This is dependent on the function this node podds.
        summa is the summation of the values of the input nodes multiplied by
        their weights'''
        summa = 0
        if self.func == 'S':  # Sigmoid Function
            for x in self.inputs:
                summa += x.getValue() * self.weights[self.inputs.index(x)]
            self.value = expit(summa)
        elif self.func == 'B':  # Bias Node Function
            self.value = 1
        else:
            self.value = 1

    def calcHiddenError(self):
        '''Calculate the Error assuming this is a hidden layer node summa is
        the summation of the errors from the output nodes multiplied by their
        weights.'''
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
            # self.error = 0.5 * (answer - self.value)**2
        else:
            self.error = 0

    def updateWeights(self, LearnRate, Momentum, loop):
        '''Update all of the input weights for this node via backprop.
        Requires the Learning Rate (LearnRate), Momentum, and current loop
        (loop) to calculate'''
        global Bloops
        # DLR = 0
        DLR = 1 - 1 / (Bloops - loop + 1)  # Linear decreasing relationship
        if self.func == 'S':
            for i in range(len(self.weights)):
                temp = self.weights[i]
                self.weights[i] += ((1 - Momentum) * max(LearnRate, DLR) * self.error * self.inputs[
                                    i].getValue()) + (Momentum * (self.weights[i] - self.historicalWeights[i]))
                self.historicalWeights[i] = temp


class NN(object):
    '''A single Neural Network that will approximate a function via an input
    vector, node arrangement matrix, output vector, answer vector, learning
    rate (optional) (0,1], threshold value (optional) (0,Infinity), momentum
    value (optional) (0,1].
    Scaling our outputs according to the domain of all our possible answers.
    New range is between 0.2 and 0.8 thus having a buffer of 0.2 on either side
    to accommodate an approach from that direction (initial guess) or the
    possibility of estimating an answer that exceeds the domain of our training
    data.'''

    def __init__(self, inputs, arrangement, outputs, answers, learnrate=0.3,
                 threshold=1, momentum=0.5, maxim=0, minim=1000):
        self.StartingNodes = []
        self.HiddenNodes = []
        self.OutputNodes = []
        self.Threshold = threshold
        self.AnswerSet = answers
        self.LearnRate = learnrate
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
                    n.addInputs(self.HiddenNodes[
                                self.arrangement.index(y) - 1])
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

    def GetStartingNodesValues(self):
        ans = []
        for i in range(len(self.StartingNodes)):
            ans.append(self.StartingNodes[i].getValue())
        return ans

    def SetAnswerSetValues(self, values):
        '''Reset the answerSet for the NN that is used to train against to
        values'''
        for i in range(len(self.AnswerSet)):
            self.AnswerSet = values
        for i in range(len(self.AnswerSet)):
            if self.maxim == self.minim:
                self.AnswerSet[i] = self.maxim / (2 * self.maxim)
            else:
                self.AnswerSet[i] = (
                    ((self.AnswerSet[i] - self.minim) * (0.8 - 0.2)) /
                    (self.maxim - self.minim)) + 0.2

    def PrintStatus(self):
        '''Print all the values, errors, and weights contained within this
        NN'''
        print()
        for x in self.StartingNodes:
            print(id(x), 'has starting value:', x.getValue())
        for y in self.HiddenNodes:
            for x in y:
                print(id(x), 'has hidden value:', x.getValue())
                print(id(x), 'has hidden error:', x.getError())
                print(id(x), 'had weights:', x.getWeights())
        for x in self.OutputNodes:
            print(id(x), 'has output value:', x.getValue(), '~',
                  self.AnswerSet[self.OutputNodes.index(x)])
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
        '''Calculate the error from the output. Should only ever be run after
        ShouldBackprop() has been run.'''
        for i in range(len(list(reversed(self.OutputNodes)))):
            self.OutputNodes[i].calcOutputError(self.AnswerSet[i])
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
        '''Returns the set of all the weights contained within the NN after
        removing the bias node's weights'''
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
        '''Sets all of the weights of the network. Mirror function for
        GetNNWeights'''
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
            self.OutputNodes[i].updateWeights(
                self.LearnRate, self.Momentum, loop)

    def GetNNResults(self):
        '''Returns the values of the output nodes'''
        resultSet = []
        for i in range(len(self.OutputNodes)):
            resultSet.append((((self.OutputNodes[i].getValue() - 0.2) *
                               (self.maxim - self.minim)) / (0.8 - 0.2)) + self.minim)
        return resultSet
        
    def GetNNResultsUnscaled(self):
        '''Returns the values of the output nodes unscaled'''
        resultSet = []
        for i in range(len(self.OutputNodes)):
            resultSet.append(self.OutputNodes[i].getValue())
        return resultSet

#    def ShouldBackprop(self, answers):
#        '''Calculates the error of the output nodes and determines if the
#        results are within the threshold percentage.'''
#        backprop = False
#        if (calcRelativeError(self, self.inputs, answers)):
#            None
#        # for i in range(len(self.OutputNodes)):
#        #     self.OutputNodes[i].calcOutputError(self.AnswerSet[i])
#        #     if 0.5 * sum(list(map(lambda i: (self.GetNNResults()[i] - answers[i])**2,
#        #                           range(len(answers))))) > self.Threshold:
#        #         backprop = True
#        self.converged = backprop
#        return self.converged


########## Doesn't currently work nor is it relavent really... ###########
#
# def d(NNset, answers):
#     '''Returns the list of errors for the output nodes.
#     This is effectively the Sum of Squares Residuals'''
#     if answers[0][0] < 1:
#         NNTesting = list(
#             map(lambda i: NNset[i].GetNNResultsUnscaled(), range(len(NNset))))
#     else:
#         NNTesting = list(
#             map(lambda i: NNset[i].GetNNResults(), range(len(NNset))))
#     errorValue = 0
#     for i in range(len(NNTesting)):
#         for j in range(len(NNTesting[i])):
#             errorValue += (NNTesting[i][j] - answers[i][j])**2
#     return errorValue
#
#
# def SST(values, average):
#     '''Sum of Squares Totals'''
#     val = 0
#     for i in range(len(values)):
#         for j in range(len(values[i])):
#             val += (values[i][j] - average)**2
#     return val
#
#
# def calcRSquared(NN, answers):
#     '''Calculates an R^2 value for a NN and set of answers'''
#     return (1 - (d([NN], answers)) / (SST(answers, sum(map(sum, answers)))))
#
##########################################################################

def calcRelativeError(NN, inputs, answers):
    '''Calculate the relative error in percent of the NN, given a set of inputs and answers'''
    NNWorking = copy.deepcopy(NN)
    #StartingWeights = NN.GetNNWeights()
    #NNWorking = NN
    count = 0
    errorValue = 0
    for i in range(len(inputs)):
        NNWorking.SetStartingNodesValues(inputs[i])
        NNWorking.CalculateNNOutputs()
        if answers[0][0] < 1: # Check if it's been scaled. 
            NNRes = NNWorking.GetNNResultsUnscaled()
        else:
            NNRes = NNWorking.GetNNResults()
        for j in range(len(NNRes)):
            errorValue += (abs(NNRes[j] - answers[i][j]) / answers[i][j])
            count += 1
    #NN.SetNNWeights(StartingWeights)
    return errorValue / count

def calcLeastSquaresError(NN, inputs, answers):
    '''Calculate the Least Squares Regression of the NN, given a set of inputs and answers'''
    NNWorking = copy.deepcopy(NN)
    errorValue = 0
    for i in range(len(inputs)):
        NNWorking.SetStartingNodesValues(inputs[i])
        NNWorking.CalculateNNOutputs()
        if answers[0][0] < 1: # Check if it's been scaled.
            NNRes = NNWorking.GetNNResultsUnscaled()
        else:
            NNRes = NNWorking.GetNNResults()
        for j in range(len(NNRes)):
            errorValue += (NNRes[j] - answers[i][j])**2
    return errorValue


def main(inputs, arrangement, outputs, answers, maxLoops, learnrate=0.5, threshold=1,
         momentum=0, printFile=False):
    '''Our Main Method that takes in the list of input vectors, the arrangement
    (topology), the list of output vectors, the list of answer vectors, the NN
    Learning Rate (learnrate), the threshold percentage (threshold), and the
    momentum scalar (momentum).
    Returns the NN that has been trained and is ready for testing.
    Testing code will be handled in the Handler File.'''
    global Bloops
    Bloops = maxLoops
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
    baseNN = NN(inputs[0], arrangement, outputs, answers[0],
                learnrate, threshold, momentum, maxim, minim)
    baseNN.ConstructNetwork()

    # Create a copy of the template and set it's inputs and answers to the
    # appropriate vectors.
    # Then saves this new NN as an instance in NNinstances
    for i in range(len(inputs)):
        temp = copy.deepcopy(baseNN)
        temp.SetStartingNodesValues(inputs[i])
        temp.SetAnswerSetValues(answers[i])
        NNinstances.append(temp)

    if printFile: f = open('NN.csv', 'w')
    finalErrorMeasure = 0
    errorMeasure = 0
    loops = 0
    while True:
        # Calculate the outputs of all instances.
        for i in range(len(NNinstances)):
            NNinstances[i].CalculateNNOutputs()
        # Calculate the output layer's error and determine if the network needs
        # to backprop.
        # Merge all the weights from every instance and average the values,
        # then reset each NN to have these new Weights
        weightSet = []
        for i in range(len(NNinstances)):
            weightSet.append(NNinstances[i].GetNNWeights())
        weightSet = transpose(weightSet)
        newWeightSet = []
        for x in weightSet:
            newWeightSet.append((sum(x)) / len(x))
        for i in range(len(NNinstances)):
            NNinstances[i].SetNNWeights(newWeightSet)
        # Calculate the output layer's error and determine if the network needs
        # to backprop again.
        # If, at this point, neither test has failed and flipped done to false,
        # we have a valid Weight set for use as a solution.
        # for i in range(len(NNinstances)):
        if calcRelativeError(NNinstances[0], inputs, OrigAnswers) * 100 < threshold:
            break
        else:
            #NNinstances[0].PrintStatus()
            print("\rTraining: {:2.2%}".format(calcRelativeError(NNinstances[
                  0], inputs, OrigAnswers)), "{:2.2%}   ".format(loops / Bloops), end="\r")
            #NNinstances[0].PrintStatus()
            if printFile: f.write('%f,' % calcRelativeError(
                NNinstances[0], inputs, OrigAnswers))
            if printFile: f.write('\n')
        loops += 1
        # Gets us out of this loop if we have backproped more times than our
        # max number of loops: Bloops
        if loops >= (Bloops):
            break
        # If we reach this point, then the network needs to fully backprop and
        # update its errors and weights before recalculating its outputs.
        for i in range(len(NNinstances)):
            NNinstances[i].CalculateNNErrors()
            NNinstances[i].UpdateNNWeights(loops)

    # Select one of the finished NN's as they should all be the same and call
    # it your final NN.
    if printFile: f.close()
    finalNN = copy.deepcopy(NNinstances[0])

    # Test your original input vectors on the finalNN. Results should be
    # accurate...
    print('Loops: %d' % loops, ' ' * 20)
    # print("Error R^2: {:2.5%}".format(calcRSquared(finalNN, inputs,
    # OrigAnswers)))
    print("Error Relative: {:2.5%}".format(calcRelativeError(finalNN, inputs, OrigAnswers)))
    print("Least Squares: %d" % calcLeastSquaresError(finalNN, inputs, OrigAnswers))
    for x in inputs:
        finalNN.SetStartingNodesValues(x)
        finalNN.CalculateNNOutputs()
        print(x, finalNN.GetNNResults(), OrigAnswers[inputs.index(x)])
    print()

    # Ready to run tests on this NN
    return finalNN

if __name__ == '__main__':
    print('Starting some NN training...\n')

    for i in range(1):
        main([[2, 3], [1, 3], [3, 3]], [['S', 'S', 'S'], ['S', 'S']], ['S'],
             [[101], [400], [3604]], maxLoops=100000, learnrate=0.3, threshold=5, momentum=0.5, printFile=False)
