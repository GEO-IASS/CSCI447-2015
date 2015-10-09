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

class node:
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

	def addInputs(self, nodes):
		for x in nodes:
			x.addOutput(self)
			self.inputs.append(x)
			self.weights.append((random.random()*2)-1)
			self.historicalWeights.append(0)
		self.inputs.append(node(appFunc = 'B', value = 1))
		self.weights.append((random.random()*2)-1)
		self.historicalWeights.append(0)

	def getInputs(self):
		return self.inputs

	def addOutput(self, node):
		self.outputs.append(node)

	def getOutputs(self):
		return self.outputs

	def setValue(self, value):
		self.value = value

	def getValue(self):
		return self.value

	def setNewError(self, newError):
		self.error = newError

	def getError(self):
		return self.error

	def getWeightForNode(self, node):
		return self.weights[self.inputs.index(node)]

	def getWeightOutputs(self):
		temp = []
		for x in self.outputs: temp.append(x.getWeightForNode(self))
		return temp

	def getWeights(self):
		return self.weights

	def setWeights(self, values):
		self.weights = values

	def setDmax(self, newdmax):
		self.dmax = newdmax
 
	def calcValue(self): 
		summa = 0	
		for x in self.inputs: summa += x.getValue()*self.weights[self.inputs.index(x)]
		if self.func == 'S':	#sigmoid
			self.value = 1 / (1 + math.exp(-summa))
		elif self.func == 'B': 	#bias value
			self.value = 1
		elif self.func == 'G': 	#guassian 
			inputVector = []
			for x in self.inputs:
				inputVector.append(x.getValue())
			gaussInput = list(map(sub, inputVector, self.weights[:-1]))
			beta = (4*len(self.inputs[0].getOutputs())/(self.dmax**2))
			self.value = math.e**(-beta*EuclideanDistance(gaussInput))
		elif self.func == 'L': 	#linear step
			if summa > 0: self.value = 1
			else: self.value = 0
		elif self.func == 'R':	#summation
			self.value = summa
		else: self.value = 1

	def calcHiddenError(self):
		summa = 0
		for x in self.outputs: summa += x.getError() * x.getWeightForNode(self)
		if self.func == 'S':
			self.error = self.value * (1-self.value) * summa
		elif self.func == 'L':
			self.error = summa 
			#self.error = self.value 
		elif self.func == 'G':
			self.error = 1

	def calcOutputError(self, answer):
		if self.func == 'S':	
			self.error = (answer - self.value) * self.value * (1 - self.value)
		elif self.func == 'L': 
			#print(answer, self.value)
			self.error = (answer - self.value)
		elif self.func == 'R':
			self.error = (answer - self.value)
			#self.error = (answer - self.value)**2 This should be the error, but the values don't converge. 
		else: self.error = 0

	def updateWeights(self, LearnRate, Momentum, loop):
		global Bloops
		DLR = 1 - 1/(Bloops-loop+1) # Linear decreasing relationship
		#DLR = -((1 + 1/(Bloops ** 5))**(loop ** 10)) + 2 # This is a bit arbitrary... 
		if self.func == 'S' or self.func == 'L' or self.func == 'R':
			for i in range(len(self.weights)):
				temp = self.weights[i]
				self.weights[i] = self.weights[i] + ((1 - Momentum) * max(LearnRate, DLR) * self.error * self.inputs[i].getValue())
				self.weights[i] = self.weights[i] + (Momentum * (self.weights[i] - self.historicalWeights[i]))
				self.historicalWeights[i] = temp
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

class NN:
	def __init__(self, inputs, arrangement, outputs, answers, learnrate, threshold, momentum):
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

	def SetDmax(self, newdmax):
		self.Dmax = newdmax
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].setDmax(newdmax)

	def ConstructNetwork(self):
		# Construct the network from the inputs
		# Make Start Nodes
		for x in self.inputs:
			n = node(value = x)
			self.StartingNodes.append(n)
		# Make Hidden Layers
		for y in self.arrangement:
			temp = []
			for x in y:
				if self.arrangement.index(y) == 0:
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

	def SetStartingNodesValues(self, values):
		for i in range(len(self.StartingNodes)):
			self.StartingNodes[i].setValue(values[i])

	def SetAnswerSetValues(self, values):
		for i in range(len(self.AnswerSet)):
			self.AnswerSet = values

	def getAnswerSet(self):
		return self.AnswerSet

	def PrintStatus(self):
		print()
		#for x in self.StartingNodes:
		#	print(id(x), 'has starting value:', x.getValue())
		#for y in self.HiddenNodes:
		#	for x in y:
		#		print(id(x), 'has hidden value:', x.getValue())
		#		print(id(x), 'has hidden error:', x.getError())
		#		print(id(x), 'had weights:', x.getWeights())
		for x in self.OutputNodes:
			print(id(x), 'has output value:', x.getValue(), '~', self.AnswerSet[self.OutputNodes.index(x)])
			#print(id(x), 'has output error:', x.getError())
			#print(id(x), 'had weights:', x.getWeights())

	def CalculateNNOutputs(self):
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].calcValue()
		for i in range(len(self.OutputNodes)):
			self.OutputNodes[i].calcValue()

	def CalculateNNErrors(self):
		for i in range(len(list(reversed(self.HiddenNodes)))):
			for j in range(len(list(reversed(self.HiddenNodes))[i])):
				(list(reversed(self.HiddenNodes))[i][j]).calcHiddenError()

	def GetNNWeights(self):
		weightSet = []
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				for x in self.HiddenNodes[i][j].getWeights(): weightSet.append(x)
		for i in range(len(self.OutputNodes)):
			for x in self.OutputNodes[i].getWeights(): 
				weightSet.append(x)
		return weightSet

	def GetNNWeightsTrim(self):
		weightSet = []
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				for x in self.HiddenNodes[i][j].getWeights()[:-1]: weightSet.append(x)
		for i in range(len(self.OutputNodes)):
			for x in self.OutputNodes[i].getWeights()[:-1]: 
				weightSet.append(x)
		return weightSet

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

	def UpdateNNWeights(self, loop):
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].updateWeights(self.LearnRate, self.Momentum, loop)
		for i in range(len(self.OutputNodes)): 
			self.OutputNodes[i].updateWeights(self.LearnRate, self.Momentum, loop)

	def GetNNResults(self):
		resultSet = []
		for i in range(len(self.OutputNodes)): 
			resultSet.append(self.OutputNodes[i].getValue())
		return resultSet

	def ShouldBackprop(self):
		backprop = False
		for i in range(len(self.OutputNodes)):
			self.OutputNodes[i].calcOutputError(self.AnswerSet[i])
		if not ((self.OutputNodes[i].getValue() <= (self.AnswerSet[i] + (self.Threshold * self.AnswerSet[i]))) and 
			(self.OutputNodes[i].getValue() >= (self.AnswerSet[i] - (self.Threshold * self.AnswerSet[i])))):
			backprop = True
		self.converged = backprop
		return self.converged

def ProcessGroupFunc(instance, loops):
	instance.CalculateNNErrors()
	instance.UpdateNNWeights(loops)

def EuclideanDistance(vector):
	#print(vector)
	return linalg.norm(vector)

	#Still need to test Linear and Sigmoid with RBFN
	#Still need to test Linear with FFNN

def main(inputs, arrangement, outputs, answers, learnrate = 0.5, threshold = 1, momentum = 0):
	global Bloops
	Bloops = 20 ** len(inputs)
	NNinstances = []
	OrigAnswers = copy.deepcopy(answers)

	maxim = 0
	for x in answers: maxim = max(maxim, max(x))

	minim = 10000
	for x in answers: minim = min(minim, min(x))

	for i in range(len(answers)):
		for j in range(len(answers[i])):
			if (maxim == minim): 
				answers[i][j] = maxim/(2*maxim)			
			else: 
				answers[i][j] = (((answers[i][j] - minim) * (0.8 - 0.2)) / (maxim - minim)) + 0.2
				threshold = (threshold / (maxim - minim))

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

	#print('Cats')
	dmax = 0
	for x in cVectors:
		for y in cVectors:
			dmax = max((EuclideanDistance(list(map(sub, x, y)))), dmax)

	for i in range(len(inputs)):
		temp = copy.deepcopy(baseNN)
		temp.SetDmax(dmax)
		temp.SetStartingNodesValues(inputs[i])
		temp.SetAnswerSetValues(answers[i])
		NNinstances.append(temp)
	#	temp.PrintStatus()

	loops = 0
	while True:
		#print()
		for i in range(len(NNinstances)): 
			NNinstances[i].CalculateNNOutputs()
			#NNinstances[i].PrintStatus()
		done = True
		for i in range(len(NNinstances)): 
			if NNinstances[i].ShouldBackprop(): done = False
		weightSet = []
		for i in range(len(NNinstances)): weightSet.append(NNinstances[i].GetNNWeights())
		weightSet = transpose(weightSet)
		newWeightSet = []
		for x in weightSet: newWeightSet.append((sum(x))/len(x))
		for i in range(len(NNinstances)): NNinstances[i].SetNNWeights(newWeightSet)
		for i in range(len(NNinstances)): 
			if NNinstances[i].ShouldBackprop(): 
				#print(i)
				done = False
		#print()
		print("Progress {:2.1%}".format(loops / Bloops), end="\r")
		if (done and (loops >= 100)): break
		loops += 1
		if loops > (Bloops):
			#print('Reached an iterative bound. Bailing!')
			break
		for i in range(len(NNinstances)):
			NNinstances[i].CalculateNNErrors()
			NNinstances[i].UpdateNNWeights(loops)

	results = answers
	for i in range(len(NNinstances)):
		for j in range(len(NNinstances[i].GetNNResults())):
			if (maxim == minim): results[i][j] = NNinstances[i].GetNNResults()[j] * maxim * 2
			else: results[i][j] = (((NNinstances[i].GetNNResults()[j] - 0.2) * (maxim - minim)) / (0.8 - 0.2)) + minim
	for i in range(len(results)): print(loops, inputs[i], results[i], OrigAnswers[i])

	finalNN = copy.deepcopy(NNinstances[0])

	for x in inputs:
		#print(x)
		finalNN.SetStartingNodesValues(x)
		finalNN.CalculateNNOutputs()
		print(loops, x, ((((finalNN.GetNNResults()[0] - 0.2) * (maxim - minim)) / (0.8 - 0.2)) + minim), OrigAnswers[inputs.index(x)])

	#finalNN.SetStartingNodesValues([3.5])
	finalNN.CalculateNNOutputs()
	#print(loops, [3.5], ((((finalNN.GetNNResults()[0] - 0.2) * (maxim - minim)) / (0.8 - 0.2)) + minim), [12.25])

	return finalNN

# Things to add
# 	For RBFN use G for hidden and R for output

if __name__== '__main__':
	print('Starting some NN tests...\n')
	
	#main([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	#main([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400]], learnrate = 0.1, threshold = 1, momentum = 0.5)
	#main([[2,3], [1,3], [3,3]], [['S','S','S'], ['S','S']], ['S'], [[101], [400], [3604]], learnrate = 0.5, threshold = 1, momentum = 0.5)
	main([[1],[2],[3],[4],[5]], [['S','S','S','S','S'], ['S','S','S']], ['S'], [[1],[4],[9],[16],[25]], learnrate = 0.5, threshold = 5, momentum = 0.3)
	#main([[1],[2],[3],[4],[5]], [['L', 'L', 'L']], ['S'], [[1],[4],[9],[16],[25]], learnrate = .5, threshold = 5, momentum = .3)
	#main([[2,3], [1,3], [3,3]], [['G','G','G']], ['R'], [[101], [400], [3604]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	#main([[2,3], [1,3]], [['G','G','G']], ['R'], [[101], [400]], learnrate = 0.5, threshold = 10, momentum = 0.5)

