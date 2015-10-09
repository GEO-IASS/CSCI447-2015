#!/usr/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	10/08/15
CSCI 447:	Project 2
"""

# Code for a neural network... Details on the horizon!!!

import sys
import math
import random
from multiprocessing import Process
from numpy import transpose, linalg
import time
import copy
from operator import sub

# Class for a single node in either network and in input, hidden, or output layer
class node:
	# Constructor for node class
	# Takes in 3 option parameters for a function, a starting value, and a dmax
	def __init__(self, appFunc = '', value = 0, dmax = 1):
		self.inputs = []
		self.weights = []
		self.outputs = []
		self.error = 0
		self.func = appFunc
		self.value = value
		self.historicalWeights = []
		self.oldLR = 0.5
		self.dmax = dmax

	# Add an array of input nodes to this node. 
	# When doing so, adds random weights for each node.
	# Also adds a bias node with value 1 and 
	def addInputs(self, nodes):
		for x in nodes:
			x.addOutput(self)
			self.inputs.append(x)
			self.weights.append(random.random()-0.5)
			self.historicalWeights.append(0)
		self.inputs.append(node(appFunc = 'B', value = 1))
		self.weights.append(random.random()-0.5)
		self.historicalWeights.append(0)

	# Add a node as an output to this node
	def addOutput(self, node):
		self.outputs.append(node)

	# Set the value of this node without calculating it
	def setValue(self, value):
		self.value = value

	# Set the error of this node without calculating it
	def setNewError(self, newError):
		self.error = newError

	# Set the weights to the values provided by the list values
	def setWeights(self, values):
		self.weights = values

	# Set the dmax of this node to the newdmax
	def setDmax(self, newdmax):
		self.dmax = newdmax

	# Returns the set of input nodes for this node
	def getInputs(self):
		return self.inputs

	# Return the set of output nodes
	def getOutputs(self):
		return self.outputs

	# Return this nodes current value
	def getValue(self):
		return self.value

	# Return the error of this node
	def getError(self):
		return self.error

	# Return the weight for a particular input node given as an input
	def getWeightForNode(self, node):
		return self.weights[self.inputs.index(node)]

	# Return the set of weights for the output nodes and this node
	def getWeightOutputs(self):
		temp = []
		for x in self.outputs: temp.append(x.getWeightForNode(self))
		return temp

	# Return the set of weights for the input nodes and this node
	def getWeights(self):
		return self.weights
 
 	# Calculate the value of this node. 
 	# This is dependent on the function this node possesses.
	# summa is the summation of the values of the input nodes multiplied by their weights
	def calcValue(self): 
		summa = 0	
		for x in self.inputs: summa += x.getValue()*self.weights[self.inputs.index(x)]
		# Sigmoid Function
		if self.func == 'S':
			self.value = 1 / (1 + math.exp(-summa))
		# Bias Node Function
		elif self.func == 'B':
			self.value = 1
		# Gaussian Function
		elif self.func == 'G':
			inputVector = []
			for x in self.inputs:
				inputVector.append(x.getValue())
			gaussInput = list(map(sub, inputVector, self.weights[:-1]))
			width = 2*self.dmax/(2*len(self.inputs[0].getOutputs()))
			self.value = math.e**((-EuclideanDistance(gaussInput)**2)/width)
		# Linear Step Function
		elif self.func == 'L':
			if summa > 0: self.value = 1
			else: self.value = 0
		# Summation Function for RBF
		elif self.func == 'R':
			self.value = summa
		else: self.value = 1

	# Calculate the Error assuming this is a hidden layer node
	# summa is the summation of the errors from the output nodes multiplied by their weights
	def calcHiddenError(self):
		summa = 0
		for x in self.outputs: summa += x.getError() * x.getWeightForNode(self)
		# Sigmoid Error
		if self.func == 'S':
			self.error = self.value * (1-self.value) * summa
		# Linear Step Error
		elif self.func == 'L':
			self.error = summa 
		# Gaussian Error (Should not be needed / used)
		elif self.func == 'G':
			self.error = 1
		# Safety
		else:
			self.error = 0

	# Calculate the Error assuming this is an output layer node
	def calcOutputError(self, answer):
		# Sigmoid Error
		if self.func == 'S':
			self.error = (answer - self.value) * self.value * (1 - self.value)
		# Linear Step Error and RBF Error (Delta Rule as derivative of the Linear Step Function is 1)
		elif self.func == 'L' or self.func == 'R': 
			self.error = (answer - self.value)
		# Safety
		else: 
			self.error = 0

	# Update all of the input weights for this node
	# Requires the Learning Rate (LearnRate), Momentum, and current loop (loop) to calculate
	def updateWeights(self, LearnRate, Momentum, loop):
		global Bloops
		DLR = 1 - 1/(Bloops-loop+1) # Linear decreasing relationship
		# Sigmoid, Linear Step, and RBF Output nodes apply a new weight based on their error. 
		if self.func == 'S' or self.func == 'L' or self.func == 'R':
			for i in range(len(self.weights)):
				temp = self.weights[i]
				self.weights[i] = self.weights[i] + ((1 - Momentum) * max(LearnRate, DLR) * self.error * self.inputs[i].getValue()) + (Momentum * (self.weights[i] - self.historicalWeights[i]))
				self.historicalWeights[i] = temp
		# Gaussian nodes apply a new weight based on the derivative of the Gaussian equation (TBD) and their error
		elif self.func == 'G':
			for i in range(len(self.weights)):
				temp = self.weights[i]
				beta1 = -(2 * len(self.inputs[0].getOutputs()) / (self.dmax**2))
				beta2 = (2 * len(self.inputs[0].getOutputs()) / (self.dmax**3))
				gaussInput = list(map(sub, (list(map(lambda x: x.getValue(), self.inputs))), self.weights))
				norm = EuclideanDistance(gaussInput)
				self.weights[i] = self.weights[i] + ((1 - Momentum) * -1 * max(LearnRate, DLR) * ( 4 * len(self.inputs[0].getOutputs()) * norm**4 * (math.e**((-norm**2)/(2 * (self.dmax/math.sqrt(2*len(self.inputs[0].getOutputs())))))))/(self.dmax**4))
				#self.weights[i] = self.weights[i] + ((1 - Momentum) * -1 * max(LearnRate, DLR) * (beta2 * norm * (math.e ** (beta1 * (norm ** 2)))) * self.weights[i])
				self.weights[i] = self.weights[i] + (Momentum * (self.weights[i] - self.historicalWeights[i]))
				self.historicalWeights[i] = temp

# A single Neural Network that will approximate a function via an input vector, node arrangement matrix, output vector, answer vector, 
# learning rate (optional) (0,1], threshold value (optional) (0,∞), momentum value (optional) (0,1]
class NN:
	# Constructor for the Neural Network
	def __init__(self, inputs, arrangement, outputs, answers, learnrate = 0.2, threshold = 1, momentum = 0.5):
		self.StartingNodes = []
		self.HiddenNodes = []
		self.OutputNodes = []
		self.Threshold = 0.01 * threshold 
		self.AnswerSet = answers
		self.LearnRate = learnrate
		self.converged = False
		self.Momentum = momentum
		self.Dmax = 1
		self.inputs = inputs
		self.arrangement = arrangement
		self.outputs = outputs

	# Construct the network from the inputs
	def ConstructNetwork(self):
		# Make Start Nodes
		for x in self.inputs:
			n = node(value = x)
			self.StartingNodes.append(n)
		# Make Hidden Layers
		for y in self.arrangement:
			temp = []
			for x in y:
				if self.arrangement.index(y) == 0:
					# Gaussian nodes need to keep track of a dmax value
					if x == 'G': 
						n = node(appFunc = x, dmax = self.Dmax)
					else: 
						n = node(appFunc = x)
					n.addInputs(self.StartingNodes)
					temp.append(n)
				else:
					n = node(appFunc = x)
					n.addInputs(self.HiddenNodes[self.arrangement.index(y) - 1])
					temp.append(n)
			self.HiddenNodes.append(temp)
		# Make Output Layers
		for x in self.outputs:
			n = node(appFunc = x)
			n.addInputs(self.HiddenNodes[-1])
			self.OutputNodes.append(n)
		# Network created and ready to function

	# Reset the starting nodes values to values
	def SetStartingNodesValues(self, values):
		for i in range(len(self.StartingNodes)):
			self.StartingNodes[i].setValue(values[i])

	# Reset the answerSet for the NN that is used to train against to values
	def SetAnswerSetValues(self, values):
		for i in range(len(self.AnswerSet)):
			self.AnswerSet = values

	# Reset the dmax of all the hidden nodes in the NN
	def SetDmax(self, newdmax):
		self.Dmax = newdmax
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].setDmax(newdmax)

	# Print all the values, errors, and weights contained within this NN
	def PrintStatus(self):
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

	# Calculate the answer of the NN
	def CalculateNNOutputs(self):
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].calcValue()
		for i in range(len(self.OutputNodes)):
			self.OutputNodes[i].calcValue()

	# Calculate the error from the output. Should only ever be run after ShouldBackprop() has been run.
	def CalculateNNErrors(self):
		for i in range(len(list(reversed(self.HiddenNodes)))):
			for j in range(len(list(reversed(self.HiddenNodes))[i])):
				(list(reversed(self.HiddenNodes))[i][j]).calcHiddenError()

	# Returns the set of all the weights contained within the NN
	def GetNNWeights(self):
		weightSet = []
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				for x in self.HiddenNodes[i][j].getWeights(): weightSet.append(x)
		for i in range(len(self.OutputNodes)):
			for x in self.OutputNodes[i].getWeights(): 
				weightSet.append(x)
		return weightSet

	# Returns the set of all the weights contained within the NN after removing the bias node's weights
	def GetNNWeightsTrim(self):
		weightSet = []
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				for x in self.HiddenNodes[i][j].getWeights()[:-1]: weightSet.append(x)
		for i in range(len(self.OutputNodes)):
			for x in self.OutputNodes[i].getWeights()[:-1]: 
				weightSet.append(x)
		return weightSet

	# Sets all of the weights of the network. Mirror function for GetNNWeights
	def SetNNWeights(self, values):
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

	# Calculate the weights of the NN by forward propagation
	def UpdateNNWeights(self, loop):
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].updateWeights(self.LearnRate, self.Momentum, loop)
		for i in range(len(self.OutputNodes)): 
			self.OutputNodes[i].updateWeights(self.LearnRate, self.Momentum, loop)

	# Returns the values of the output nodes
	def GetNNResults(self):
		resultSet = []
		for i in range(len(self.OutputNodes)): 
			resultSet.append(self.OutputNodes[i].getValue())
		return resultSet

	# Calculates the error of the output nodes and determines if the results are within the threshold percentage. 
	def ShouldBackprop(self):
		backprop = False
		for i in range(len(self.OutputNodes)):
			self.OutputNodes[i].calcOutputError(self.AnswerSet[i])
			if not ((self.OutputNodes[i].getValue() <= (self.AnswerSet[i] + (self.Threshold * self.AnswerSet[i]))) and 
				(self.OutputNodes[i].getValue() >= (self.AnswerSet[i] - (self.Threshold * self.AnswerSet[i])))):
				backprop = True
		self.converged = backprop
		return self.converged

# Calculate the Euclidean Distance of the vector. This is mostly for easy of naming in other functions. 
def EuclideanDistance(vector):
	#print(vector)
	return linalg.norm(vector)

	#Still need to test Linear and Sigmoid with RBFN
	#Still need to test Linear with FFNN

# Our Main Method that takes in the list of input vectors, the arrangement (topology), the list of output vectors, the list of answer vectors,
# the NN Learning Rate (learnrate), the threshold percentage (threshold), and the momentum scalar (momentum)
# Returns the NN that has been trained and is ready for testing. Testing code will be handled in the Handler File.
def main(inputs, arrangement, outputs, answers, learnrate = 0.5, threshold = 1, momentum = 0):
	global Bloops
	Bloops = 100 ** len(inputs)
	NNinstances = []
	OrigAnswers = copy.deepcopy(answers)

	# Max and Min of our outputs
	maxim = 0
	for x in answers: maxim = max(maxim, max(x))
	minim = 10000
	for x in answers: minim = min(minim, min(x))

	# Scale our outputs and threshold according to the domain of all our possible answers
	# New range is between 0.2 and 0.8 thus having a buffer of 0.2 on either side to accommodate an approach from that direction (initial guess)
	# or the possibility of estimating an answer that exceeds the domain of our training data. 
	for i in range(len(answers)):
		for j in range(len(answers[i])):
			if (maxim == minim): 
				answers[i][j] = maxim/(2*maxim)			
			else: 
				answers[i][j] = (((answers[i][j] - minim) * (0.8 - 0.2)) / (maxim - minim)) + 0.2
				threshold = (threshold / (maxim - minim))

	# Initial NN template that is duplicated for each input vector. 
	baseNN = NN(inputs[0], arrangement, outputs, answers[0], learnrate, threshold, momentum)
	baseNN.ConstructNetwork()

	cVectors = []
	counter = 1
	temp = []
	for x in baseNN.GetNNWeightsTrim()[:(len(inputs[0])*len(arrangement[0]))]:
		if counter%2 == 0: 
			temp.append(x)
			cVectors.append(temp)
			temp = []
		else: 
			temp.append(x)
		counter += 1

	# Calculate the max spread between estimated centers. This calculation may need a bit of fine tuning. 
	dmax = 0
	for x in cVectors:
		for y in cVectors:
			dmax = max((EuclideanDistance(list(map(sub, x, y)))), dmax)

	# Create a copy of the template and set it's inputs and answers to the appropriate vectors.
	# Then saves this new NN as an instance in NNinstances
	for i in range(len(inputs)):
		temp = copy.deepcopy(baseNN)
		temp.SetDmax(dmax)
		temp.SetStartingNodesValues(inputs[i])
		temp.SetAnswerSetValues(answers[i])
		NNinstances.append(temp)
	#	temp.PrintStatus()

	loops = 0
	while True:
		# Calculate the outputs of all instances.
		for i in range(len(NNinstances)): 
			NNinstances[i].CalculateNNOutputs()
		# Calculate the output layer's error and determine if the network needs to backprop.
		done = True
		for i in range(len(NNinstances)): 
			if NNinstances[i].ShouldBackprop(): done = False
		# Merge all the weights from every instance and average the values, then reset each NN to have these new Weights
		weightSet = []
		for i in range(len(NNinstances)): weightSet.append(NNinstances[i].GetNNWeights())
		weightSet = transpose(weightSet)
		newWeightSet = []
		for x in weightSet: newWeightSet.append((sum(x))/len(x))
		for i in range(len(NNinstances)): NNinstances[i].SetNNWeights(newWeightSet)
		# Calculate the output layer's error and determine if the network needs to backprop again. 
		# If at this point, neither test has failed and flipped done to false, we have a valid Weight set for use as a solution.
		for i in range(len(NNinstances)): 
			if NNinstances[i].ShouldBackprop(): 
				done = False
		print("Progress {:2.1%}".format(loops / Bloops), end="\r")
		# Make sure we have iterated at least 100 times before presenting our solution. Prevents us from being lucky.
		if (done and (loops >= 100)): break
		loops += 1
		# Gets us out of this loop if we have backproped more times than our Max number of loops: Bloops
		if loops > (Bloops):
			#print('Reached an iterative bound. Bailing!')
			break
		# If we reach this point, then the network needs to fully backprop and update its errors and weights before recalculating its outputs.
		for i in range(len(NNinstances)):
			NNinstances[i].CalculateNNErrors()
			NNinstances[i].UpdateNNWeights(loops)

	# See the output of each NN and how close it thought it got to the function it was learning. 
	results = answers
	for i in range(len(NNinstances)):
		for j in range(len(NNinstances[i].GetNNResults())):
			if (maxim == minim): results[i][j] = NNinstances[i].GetNNResults()[j] * maxim * 2
			else: results[i][j] = (((NNinstances[i].GetNNResults()[j] - 0.2) * (maxim - minim)) / (0.8 - 0.2)) + minim
	for i in range(len(results)): print(loops, inputs[i], results[i], OrigAnswers[i])

	# Select one of the finished NN's as they should all be the same and call it your final NN. 
	finalNN = copy.deepcopy(NNinstances[0])

	# Test your original input vectors on the finalNN. Results should be very accurate. 
	for x in inputs:
		#print(x)
		finalNN.SetStartingNodesValues(x)
		finalNN.CalculateNNOutputs()
		print(loops, x, ((((finalNN.GetNNResults()[0] - 0.2) * (maxim - minim)) / (0.8 - 0.2)) + minim), OrigAnswers[inputs.index(x)])

	# Testing Example
	#finalNN.SetStartingNodesValues([3.5])
	#finalNN.CalculateNNOutputs()
	#print(loops, [3.5], ((((finalNN.GetNNResults()[0] - 0.2) * (maxim - minim)) / (0.8 - 0.2)) + minim), [12.25])

	# Ready to run tests on this NN
	return finalNN

if __name__== '__main__':
	print('Starting some NN tests...\n')
	
	#main([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	#main([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400]], learnrate = 0.1, threshold = 1, momentum = 0.5)
	#main([[2,3], [1,3], [3,3]], [['S','S','S'], ['S','S']], ['S'], [[101], [400], [3604]], learnrate = 0.5, threshold = 1, momentum = 0.5)
	#main([[1],[2],[3],[4],[5]], [['S','S','S','S','S'], ['S','S','S']], ['S'], [[1],[4],[9],[16],[25]], learnrate = 0.5, threshold = 5, momentum = 0.3)
	#main([[1],[2],[3],[4],[5]], [['L', 'L', 'L']], ['S'], [[1],[4],[9],[16],[25]], learnrate = .5, threshold = 5, momentum = .3)
	#main([[2,3], [1,3], [3,3]], [['G','G','G']], ['R'], [[101], [400], [3604]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	main([[2,3], [1,3]], [['G','G','G']], ['R'], [[101], [400]], learnrate = 0.5, threshold = 10, momentum = 0.5)

