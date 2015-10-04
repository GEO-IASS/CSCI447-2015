#!/usr/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	10/03/15
CSCI 447:	Project 2
"""

# Code for a neural network... Details on the horizon!!!

import sys
import math
import random
from multiprocessing import Process
from numpy import transpose
import time

class node:
	def __init__(self, appFunc = '', value = 0):
		self.inputs = []
		self.weights = []
		self.outputs = []
		self.error = 0
		self.func = appFunc
		self.value = value

	def addInputs(self, nodes):
		for x in nodes:
			x.addOutput(self)
			self.inputs.append(x)
			self.weights.append((random.random()*2)-1)
	def addOutput(self, node):
		self.outputs.append(node)
	def getValue(self):
		return self.value
	def getError(self):
		return self.error
	def getWeightForNode(self, node):
		return self.weights[self.inputs.index(node)]
	def getWeightOutputs(self):
		temp = []
		for x in self.outputs:
			temp.append(x.getWeightForNode(self))
		return temp
	def getOutputs(self):
		return self.outputs
	def getWeights(self):				#Temp
		return self.weights
	def getInputs(self):
		return self.inputs
	def setNewError(self, newError):
		self.error = newError
 
	def calcValue(self, BWeight): 
		summa = 0
		for x in self.inputs: summa += x.getValue()*self.weights[self.inputs.index(x)]
		summa += BWeight
	#	if self.func == 'S': self.value = expit(summa)
		if self.func == 'S': self.value = 1 / (1 + math.exp(-summa))
	#	if self.func == 'S': self.value = .5 * (1 + math.tanh(.5*summa))
		else: self.value = summa
	def calcHiddenError(self):
		summa = 0
		for x in self.outputs: summa += x.getError() * x.getWeightForNode(self)
		self.error = self.value * (1-self.value) * summa
	def calcOutputError(self, answer):
		self.error = (answer - self.value) * self.value * (1 - self.value)

	def updateWeights(self, LearnRate):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] + (LearnRate * self.error * self.inputs[i].getValue())
#	def updateWeights(self):
#		if momen == 'M': 
#			for i in range(len(self.weights)):
#				self.weights[i] = Momentum * self.weights[i] + (LearnRate * self.error * self.inputs[i].getValue())
#		else: 
#			for i in range(len(self.weights)):
#				self.weights[i] = self.weights[i] + (LearnRate * self.error * self.inputs[i].getValue())

class NN:
	def __init__(self, inputs, arrangement, outputs, answers, threshold = 1, learnrate = 0.5, bias = 1):
		self.StartingNodes = []
		self.HiddenNodes = []
		self.OutputNodes = []
		self.Threshold = 0.01 * threshold
		self.BWeight = bias
		self.AnswerSet = answers
		self.loops = 0
		self.LearnRate = learnrate
		self.converged = False

		# Construct the network from the inputs
		# Make Start Nodes
		for x in inputs:
			n = node(value = x)
			self.StartingNodes.append(n)
		# Make Hidden Layers
		for y in arrangement:
			temp = []
			for x in y:
				if arrangement.index(y) == 0:
					n = node(appFunc = x)
					n.addInputs(self.StartingNodes)
					temp.append(n)
				else:
					n = node(appFunc = x)
					n.addInputs(self.HiddenNodes[arrangement.index(y) - 1])
					temp.append(n)
			self.HiddenNodes.append(temp)
		# Make Output Layers
		for x in outputs:
			n = node(appFunc = x)
			n.addInputs(self.HiddenNodes[-1])
			self.OutputNodes.append(n)
		# Network created and ready to function
	def printStatus(self):
		print('\nWeights Found in', self.loops, 'iterations.')
		for x in self.StartingNodes:
			print(id(x), 'has starting value:', x.getValue())
		for y in self.HiddenNodes:
			for x in y:
				print(id(x), 'has hidden value:', x.getValue())
				print(id(x), 'has hidden error:', x.getError())
				print(id(x), 'had weights:', x.getWeights())
		for x in self.OutputNodes:
			print(id(x), 'has output value:', x.getValue(), '~', self.AnswerSet[OutputNodes.index(x)])
			print(id(x), 'has output error:', x.getError())
			print(id(x), 'had weights:', x.getWeights())
	def CalculateNNOutputs(self):
		backprop = False
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].calcValue(self.BWeight)
				#print('Hidden Value of', id(self.HiddenNodes[i][j]), 'is', self.HiddenNodes[i][j].getValue(), 'with weights:', self.HiddenNodes[i][j].getWeights())
		for i in range(len(self.OutputNodes)):
			self.OutputNodes[i].calcValue(self.BWeight)
			#print('Output Value of', id(self.OutputNodes[i]), self.OutputNodes[i].getValue(), 'with weights:', self.OutputNodes[i].getWeights())
			self.OutputNodes[i].calcOutputError(self.AnswerSet[i])
			#print('Output Error of', id(self.OutputNodes[i]), self.OutputNodes[i].getError())
			if not ((self.OutputNodes[i].getValue() <= (self.AnswerSet[i] + (self.Threshold * self.AnswerSet[i]))) and (self.OutputNodes[i].getValue() >= (self.AnswerSet[i] - (self.Threshold * self.AnswerSet[i])))):
				backprop = True
			self.converged = backprop
	def CalculateNNErrors(self):
		for i in range(len(list(reversed(self.HiddenNodes)))):
			for j in range(len(list(reversed(self.HiddenNodes))[i])):
				(list(reversed(self.HiddenNodes))[i][j]).calcHiddenError()
	def GetNNErrors(self):
		errorSet = []
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				errorSet.append(self.HiddenNodes[i][j].getError())
		for i in range(len(self.OutputNodes)):
			errorSet.append(self.OutputNodes[i].getError())
		return errorSet
	def SetNNErrors(self, errorSet):
		#print(errorSet)
		counter = 0
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].setNewError(errorSet[counter])
				#print(counter)
				counter += 1
		for i in range(len(self.OutputNodes)):
			self.OutputNodes[i].setNewError(errorSet[counter])
			counter += 1	
	def UpdateNNWeights(self):
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].updateWeights(self.LearnRate)
		for i in range(len(self.OutputNodes)):
			self.OutputNodes[i].updateWeights(self.LearnRate)
	def GetNNResults(self):
		resultSet = []
		for i in range(len(self.OutputNodes)):
			resultSet.append(self.OutputNodes[i].getValue())
		return resultSet
	def ShouldBackprop(self):
		return self.converged

#def ProcessNN(instanceSet, targetFunc):
#	NNprocesses = []
#	for i in range(len(instanceSet)):
#		NNprocesses.append(Process(target=instanceSet[i].targetFunc()))
#		NNprocesses[i].start()
#	for i in range(len(NNprocesses)):
#		NNprocesses[i].join()

def mainParallel(inputs, arrangement, outputs, answers, learnrate = 0.5, threshold = 1, bias = 1):
	NNinstances = []

	for i in range(len(inputs)):
		NNinstances.append(NN(inputs[i], arrangement, outputs, answers[i], learnrate, threshold, bias))

	loops = 0
	while True:
		#ProcessNN(NNinstances, CalculateNNOutputs)
		NNprocesses = []
		for i in range(len(NNinstances)):
			NNprocesses.append(Process(target=NNinstances[i].CalculateNNOutputs()))
			NNprocesses[i].start()
		for i in range(len(NNprocesses)):
			NNprocesses[i].join()
		done = True
		for i in range(len(NNinstances)):
			if NNinstances[i].ShouldBackprop():
				done = False
		#	else:
		#		print(i, NNinstances[i].GetNNResults())
		if done:
			break # All NNs have converged
		loops += 1
		if loops > (10000000 * len(inputs)):
			print('Reached an iterative bound. Bailing!')
			break
		#ProcessNN(NNinstances, CalculateNNErrors)
		NNprocesses = []
		for i in range(len(NNinstances)):
			NNprocesses.append(Process(target=NNinstances[i].CalculateNNErrors()))
			NNprocesses[i].start()
		for i in range(len(NNprocesses)):
			NNprocesses[i].join()
		errorSet = []
		for i in range(len(NNinstances)):
			errorSet.append(NNinstances[i].GetNNErrors())
		#print('ErrorSet:\n', errorSet)
		#zip(*errorSet) # Transpose Matrix
		errorSet = transpose(errorSet)
		newErrorSet = []
		for x in errorSet:
			newErrorSet.append((sum(x))/len(x))
		#for i in range(len(NNinstances)):
		#	NNinstances[i].SetNNErrors(newErrorSet)
		#print('ErrorSet:\n', errorSet)
		#print('NewErrorSet:\n', newErrorSet)
		NNprocesses = []
		for i in range(len(NNinstances)):
			NNprocesses.append(Process(target=NNinstances[i].SetNNErrors(newErrorSet)))
			NNprocesses[i].start()
		for i in range(len(NNprocesses)):
			NNprocesses[i].join()
		#ProcessNN(NNinstances, UpdateNNWeights())
		NNprocesses = []
		for i in range(len(NNinstances)):
			NNprocesses.append(Process(target=NNinstances[i].UpdateNNWeights()))
			NNprocesses[i].start()
		for i in range(len(NNprocesses)):
			NNprocesses[i].join()

	for i in range(len(NNinstances)):
		print(i, NNinstances[i].GetNNResults())

def mainIterative(inputs, arrangement, outputs, answers, learnrate = 0.5, threshold = 1, bias = 1):
	NNinstances = []

	for i in range(len(inputs)):
		NNinstances.append(NN(inputs[i], arrangement, outputs, answers[i], learnrate, threshold, bias))

	loops = 0
	while True:
		for i in range(len(NNinstances)):
			NNinstances[i].CalculateNNOutputs()
		done = True
		for i in range(len(NNinstances)):
			if NNinstances[i].ShouldBackprop():
				done = False
		#	else:
		#		print(i, NNinstances[i].GetNNResults())
		if done:
			break # All NNs have converged
		loops += 1
		if loops > (10000000 * len(inputs)):
			print('Reached an iterative bound. Bailing!')
			break
		for i in range(len(NNinstances)):
			target=NNinstances[i].CalculateNNErrors()
		errorSet = []
		for i in range(len(NNinstances)):
			errorSet.append(NNinstances[i].GetNNErrors())
		errorSet = transpose(errorSet)
		newErrorSet = []
		for x in errorSet:
			newErrorSet.append((sum(x))/len(x))
		for i in range(len(NNinstances)):
			target=NNinstances[i].SetNNErrors(newErrorSet)
		NNprocesses = []
		for i in range(len(NNinstances)):
			target=NNinstances[i].UpdateNNWeights()

	for i in range(len(NNinstances)):
		print(i, NNinstances[i].GetNNResults())

# This is a testing set. Build looks like:
	#
	#   # - A - D 
	#     \   /   \
	#       B       F - #
	#     /   \   /
	#   # - C - E
	#

if __name__== '__main__':
	print('Starting some NN tests...')

	start = time.time()
	for i in range(3): mainParallel([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101]], 0.5, 25, 1)
	end = time.time()
	print('One Set ~ Parallel ~ Average Time:', (end - start)/3)
	print()
	start = time.time()
	for i in range(3): mainIterative([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101]], 0.5, 25, 1)
	end = time.time()
	print('One Set ~ Iterative ~ Average Time:', (end - start)/3)
	print()
	start = time.time()
	for i in range(3): mainParallel([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400]], 0.5, 25, 1)
	end = time.time()
	print('Two Set ~ Parallel ~ Average Time:', (end - start)/3)
	print()
	start = time.time()
	for i in range(3): mainIterative([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400]], 0.5, 25, 1)
	end = time.time()
	print('Two Set ~ Iterative ~ Average Time:', (end - start)/3)
	print()
	start = time.time()
	for i in range(3): mainParallel([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400], [0.3604]], 0.5, 25, 1)
	end = time.time()
	print('Three Set ~ Parallel ~ Average Time:', (end - start)/3)
	print()
	start = time.time()
	for i in range(3): mainIterative([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400], [0.3604]], 0.5, 25, 1)
	end = time.time()
	print('Three Set ~ Iterative ~ Average Time:', (end - start)/3)


#if __name__== '__main__': mainParallel([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101]], 0.5, 10, 1) #Takes roughly 32 secs
#if __name__== '__main__': mainIterative([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101]], 0.5, 10, 1) #Takes roughly 0.5 secs

#if __name__== '__main__': mainParallel([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400]], 0.5, 10, 1) #Ran 10130 secs without answer...
#if __name__== '__main__': mainIterative([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400]], 0.5, 10, 1) #Takes roughly 30 secs

#if __name__== '__main__': mainParallel([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400], [0.3604]], 0.5, 10, 1)
#if __name__== '__main__': mainIterative([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400], [0.3604]], 0.5, 10, 1)


