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
from scipy.special import expit
from multiprocessing import Process
from numpy import transpose
import time
import copy #copy.deepcopy(x)

global Bloops
Bloops = 10000

class node:
	def __init__(self, appFunc = '', value = 0):
		self.inputs = []
		self.weights = []
		self.outputs = []
		self.error = 0
		self.func = appFunc
		self.value = value
		self.historicalWeights = []
		self.oldLR = 0.5

	def addInputs(self, nodes):
		for x in nodes:
			x.addOutput(self)
			self.inputs.append(x)
			self.weights.append((random.random()*2)-1)
			self.historicalWeights.append(0)
		self.inputs.append(node('B', 1))
		self.weights.append((random.random()*2)-1)
		self.historicalWeights.append(0)
	def addOutput(self, node):
		self.outputs.append(node)
	def setValue(self, value):
		self.value = value
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
	def setWeights(self, values):
		self.weights = values
	def getInputs(self):
		return self.inputs
	def setNewError(self, newError):
		self.error = newError
 
	def calcValue(self, loop): 
		if self.func == 'S':
			summa = 0
			for x in self.inputs: summa += x.getValue()*self.weights[self.inputs.index(x)]
			self.value = expit(summa)
	#		self.value = 1 / (1 + math.exp(-summa))
	#		self.value = .5 * (1 + math.tanh(.5*summa))
		elif self.func == 'B': self.value = 1
		elif self.func == 'L': 
			summa = 0
			for x in self.inputs: summa += x.getValue()*self.weights[self.inputs.index(x)]
			self.value = (lambda x: 1 if x > 0 else 0, summa)
		else: self.value = 1
		#if loop%500 == 0: self.value = self.value + ((random.random()*2)-1)/10
	def calcHiddenError(self):
		summa = 0
		for x in self.outputs: summa += x.getError() * x.getWeightForNode(self)
		if self.func == 'S':
			self.error = self.value * (1-self.value) * summa
		elif self.func == 'L':
			self.error = summa
	def calcOutputError(self, answer, loop):
		if self.func == 'S':
			self.error = (answer - self.value) * self.value * (1 - self.value)
			#noise = 0
			#if loop%1000 == 0:
			#	if self.error > 0: noise = random.random()
			#	else: noise = -1 * random.random()
			#print('Error:', self.error)
			#self.error = self.error + noise
			#print('Error with Noise:', self.error, noise)
		elif self.func == 'L':
			self.error = (answer - self.value)
		else:
			self.error = 0

	#def updateWeights(self, LearnRate):
	#	for i in range(len(self.weights)):
	#		self.weights[i] = self.weights[i] + (LearnRate * self.error * self.inputs[i].getValue())
	def updateWeights(self, LearnRate, Momentum, loop):
		global Bloops
		for i in range(len(self.weights)):
			temp = self.weights[i]
			DLR = -((1 + 1/(Bloops ** 10))**(loop ** 10)) + 2
			self.weights[i] = self.weights[i] + ((1 - Momentum) * max(LearnRate, DLR) * self.error * self.inputs[i].getValue()) + (Momentum * (self.weights[i] - self.historicalWeights[i]))
			#DLR = (((1/self.error) + 1000) * 2 / 2000) - 1 + self.oldLR
			#print(loop, i, self.error, DLR)
			#self.weights[i] = self.weights[i] + ((1 - Momentum) * DLR * self.error * self.inputs[i].getValue()) + (Momentum * (self.weights[i] - self.historicalWeights[i]))
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
		for y in self.HiddenNodes:
			for x in y:
				print(id(x), 'has hidden value:', x.getValue())
				print(id(x), 'has hidden error:', x.getError())
				#print(id(x), 'had weights:', x.getWeights())
		for x in self.OutputNodes:
			print(id(x), 'has output value:', x.getValue(), '~', self.AnswerSet[self.OutputNodes.index(x)])
			print(id(x), 'has output error:', x.getError())
			#print(id(x), 'had weights:', x.getWeights())
	def CalculateNNOutputs(self, loop):
		backprop = False
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				self.HiddenNodes[i][j].calcValue(loop)
				#print('Hidden Value of', id(self.HiddenNodes[i][j]), 'is', self.HiddenNodes[i][j].getValue(), 'with weights:', self.HiddenNodes[i][j].getWeights())
		for i in range(len(self.OutputNodes)):
			self.OutputNodes[i].calcValue(loop)
			#print('Output Value of', id(self.OutputNodes[i]), self.OutputNodes[i].getValue(), 'with weights:', self.OutputNodes[i].getWeights())
			self.OutputNodes[i].calcOutputError(self.AnswerSet[i], loop)
			#print('Output Error of', id(self.OutputNodes[i]), self.OutputNodes[i].getError())
			#print(self.OutputNodes[i].getValue(), self.AnswerSet[i] + (self.AnswerSet[i] * self.Threshold), self.AnswerSet[i] - (self.AnswerSet[i] * self.Threshold))
			if not ((self.OutputNodes[i].getValue() <= (self.AnswerSet[i] + (self.Threshold * self.AnswerSet[i]))) and (self.OutputNodes[i].getValue() >= (self.AnswerSet[i] - (self.Threshold * self.AnswerSet[i])))):
			#if not ((self.OutputNodes[i].getValue() < self.AnswerSet[i] + self.Threshold) and (self.OutputNodes[i].getValue() > self.AnswerSet[i] - self.Threshold)):
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
	def GetNNWeights(self):
		weightSet = []
		for i in range(len(self.HiddenNodes)):
			for j in range(len(self.HiddenNodes[i])):
				for x in self.HiddenNodes[i][j].getWeights():
					weightSet.append(x)
		for i in range(len(self.OutputNodes)):
			for x in self.OutputNodes[i].getWeights():
				weightSet.append(x)
		return weightSet
	def SetNNWeights(self, values):
		weightSet = []
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
		return self.converged

#def ProcessNN(instanceSet, targetFunc):
#	NNprocesses = []
#	for i in range(len(instanceSet)):
#		NNprocesses.append(Process(target=instanceSet[i].targetFunc()))
#		NNprocesses[i].start()
#	for i in range(len(NNprocesses)):
#		NNprocesses[i].join()

def mainParallel(inputs, arrangement, outputs, answers, learnrate = 0.5, threshold = 1, momentum = 0):
	global Bloops
	NNinstances = []

	maxim = 0
	for x in answers: maxim = max(maxim, max(x))

	minim = 10000
	for x in answers: minim = min(minim, min(x))


	for i in range(len(answers)):
		for j in range(len(answers[i])):
			if (maxim == minim): answers[i][j] = maxim/(2*maxim)
			else: answers[i][j] = (((answers[i][j] - minim) * (0.8 - 0.2)) / (maxim - minim)) + 0.2

	for i in range(len(inputs)):
		NNinstances.append(NN(inputs[i], arrangement, outputs, answers[i], learnrate, threshold, momentum))


	loops = 0
	while True:
		#ProcessNN(NNinstances, CalculateNNOutputs)
		NNprocesses = []
		for i in range(len(NNinstances)):
			NNprocesses.append(Process(target=NNinstances[i].CalculateNNOutputs(loops)))
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
		if loops > (Bloops):
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
			NNprocesses.append(Process(target=NNinstances[i].UpdateNNWeights(loops)))
			NNprocesses[i].start()
		for i in range(len(NNprocesses)):
			NNprocesses[i].join()

	results = answers

	for i in range(len(NNinstances)):
		for j in range(len(NNinstances[i].GetNNResults())):
			if (maxim == minim): results[i][j] = NNinstances[i].GetNNResults()[j] * maxim * 2
			else: results[i][j] = (((NNinstances[i].GetNNResults()[j] - 0.2) * (maxim - minim)) / (0.8 - 0.2)) + minim

	for i in range(len(results)):
		print(loops, i, results[i], answers[i])

def mainIterative(inputs, arrangement, outputs, answers, learnrate = 0.5, threshold = 1, momentum = 0):
	global Bloops
	NNinstances = []

	maxim = 0
	for x in answers: maxim = max(maxim, max(x))

	minim = 10000
	for x in answers: minim = min(minim, min(x))

	for i in range(len(answers)):
		for j in range(len(answers[i])):
			if (maxim == minim): answers[i][j] = maxim/(2*maxim)
			else: answers[i][j] = (((answers[i][j] - minim) * (0.8 - 0.2)) / (maxim - minim)) + 0.2

	baseNN = NN(inputs[0], arrangement, outputs, answers[0], learnrate, threshold, momentum)

	for i in range(len(inputs)):
		temp = copy.deepcopy(baseNN)
		temp.SetStartingNodesValues(inputs[i])
		temp.SetAnswerSetValues(answers[i])
		NNinstances.append(temp)

	loops = 0
	while True:
		for i in range(len(NNinstances)):
			NNinstances[i].CalculateNNOutputs(loops)
			#NNinstances[i].PrintStatus()
		done = True
		for i in range(len(NNinstances)):
			if NNinstances[i].ShouldBackprop(): done = False
			#print(i, NNinstances[i].GetNNResults())
		if done:
			weightSet = []
			for i in range(len(NNinstances)):
				weightSet.append(NNinstances[i].GetNNWeights())
			#print('Before:', weightSet)
			weightSet = transpose(weightSet)
			#print('After:', weightSet)
			newWeightSet = []
			for x in weightSet:
				newWeightSet.append((sum(x))/len(x))
			#print('New:', newWeightSet)
			for i in range(len(NNinstances)):
				NNinstances[i].SetNNWeights(newWeightSet)
		for i in range(len(NNinstances)):
			if NNinstances[i].ShouldBackprop(): done = False
			#print(i, NNinstances[i].GetNNResults())
		if done:
			break
		loops += 1
		#print(loops)
		if loops > (Bloops):
			print('Reached an iterative bound. Bailing!')
			break
		for i in range(len(NNinstances)):
			NNinstances[i].CalculateNNErrors()
		#errorSet = []
		#for i in range(len(NNinstances)):
		#	errorSet.append(NNinstances[i].GetNNErrors())
		#errorSet = transpose(errorSet)
		#newErrorSet = []
		#for x in errorSet:
		#	newErrorSet.append((sum(x))/len(x))
		#for i in range(len(NNinstances)):
		#	NNinstances[i].SetNNErrors(newErrorSet)
		for i in range(len(NNinstances)):
			NNinstances[i].UpdateNNWeights(loops)

	results = answers

	for i in range(len(NNinstances)):
		for j in range(len(NNinstances[i].GetNNResults())):
			if (maxim == minim): results[i][j] = NNinstances[i].GetNNResults()[j] * maxim * 2
			else: results[i][j] = (((NNinstances[i].GetNNResults()[j] - 0.2) * (maxim - minim)) / (0.8 - 0.2)) + minim

	for i in range(len(results)):
		print(loops, i, results[i])

# This is a testing set. Build looks like:
	#
	#   # - A - D 
	#     \   /   \
	#       B       F - #
	#     /   \   /
	#   # - C - E
	#

# Things to add
#	Dynamic learn rate (Possibly a logarithmic function)
#	Turbulence (noise) injection at # of loops to encourage new errors
# 	RBFN stuff... Leah!

if __name__== '__main__':
	print('Starting some NN tests...')
	
	#mainIterative([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	#mainIterative([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400]], learnrate = 0.5, threshold = 1, momentum = 0.5)
	mainIterative([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400], [3604]], learnrate = 0.5, threshold = 1, momentum = 0.5)
	'''
	start = time.time()
	for i in range(3): mainParallel([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	end = time.time()
	print('One Set ~ Parallel ~ Average Time:', (end - start)/3)
	print()
	start = time.time()
	for i in range(3): mainIterative([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	end = time.time()
	print('One Set ~ Iterative ~ Average Time:', (end - start)/3)
	print()
	
	#start = time.time()
	#for i in range(3): mainParallel([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	#end = time.time()
	#print('Two Set ~ Parallel ~ Average Time:', (end - start)/3)
	#print()
	start = time.time()
	for i in range(3): mainIterative([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400]], learnrate = 0.5, threshold = 10, momentum = 0.5)
	end = time.time()
	print('Two Set ~ Iterative ~ Average Time:', (end - start)/3)
	print()
	
	#start = time.time()
	#for i in range(3): mainParallel([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400], [3604]], learnrate = 0.5, threshold = 10, momentum = 0.25)
	#end = time.time()
	#print('Three Set ~ Parallel ~ Average Time:', (end - start)/3)
	#print()
	#start = time.time()
	#for i in range(3): mainIterative([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[101], [400], [3604]], learnrate = 0.75, threshold = 10, momentum = 0.75)
	#end = time.time()
	#print('Three Set ~ Iterative ~ Average Time:', (end - start)/3)
	'''

#if __name__== '__main__': mainParallel([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101]], 0.5, 10, 1) #Takes roughly 32 secs
#if __name__== '__main__': mainIterative([[2,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101]], 0.5, 10, 1) #Takes roughly 0.5 secs

#if __name__== '__main__': mainParallel([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400]], 0.5, 10, 1) #Ran 10130 secs without answer...
#if __name__== '__main__': mainIterative([[2,3], [1,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400]], 0.5, 10, 1) #Takes roughly 30 secs

#if __name__== '__main__': mainParallel([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400], [0.3604]], 0.5, 10, 1)
#if __name__== '__main__': mainIterative([[2,3], [1,3], [3,3]], [['S','S','S'], ['S', 'S']], ['S'], [[0.0101], [0.0400], [0.3604]], 0.5, 10, 1)


