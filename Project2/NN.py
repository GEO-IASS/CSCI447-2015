#!/usr/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	09/28/15
CSCI 447:	Project 2
"""

# Code for a neural network... Details on the horizon!!!

import sys
import math
import random
#from scipy.special import expit

global BWeight
BWeight = 1
global LearnRate
LearnRate = 0

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

	def calcValue(self):
		summa = 0
		for x in self.inputs: summa += x.getValue()*self.weights[self.inputs.index(x)]
		summa += BWeight
#		if self.func == 'S': self.value = expit(summa)
		if self.func == 'S': self.value = 1 / (1 + math.exp(-summa))
		else: self.value = summa
	def calcHiddenError(self):
		sigma = 0
		for x in self.outputs: sigma += x.getError() * x.getWeightForNode(self)
		self.error = self.value * (1-self.value) * sigma
	def calcOutputError(self, answer):
		self.error = (answer - self.value)
		#Technically this should be self.error = self.value * (1-self.value) * (answer - self.value)

	def updateHiddenWeights(self):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] + (LearnRate * self.error * self.weights[i])
	def updateOutputWeights(self):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] + (LearnRate * self.error * self.inputs[i].getValue())
	
def main(inputs, arrangement, outputs, answers, learnrate = 0.5, threshold = 0, bias = 0):
	global StartingNodes
	StartingNodes = []
	global HiddenNodes
	HiddenNodes = []
	global OutputNodes
	OutputNodes = []
	global Threshold
	Threshold = threshold
	global BWeight
	BWeight = bias
	global AnswerSet
	AnswerSet = answers
	global loops
	loops = 0
	global LearnRate
	LearnRate = learnrate

	# Make Start Nodes
	for x in inputs:
		n = node(value = x)
		StartingNodes.append(n)
	# Make Hidden Layers
	for y in arrangement:
		temp = []
		for x in y:
			if arrangement.index(y) == 0:
				n = node(appFunc = x)
				n.addInputs(StartingNodes)
				temp.append(n)
			else:
				n = node(appFunc = x)
				n.addInputs(HiddenNodes[arrangement.index(y) - 1])
				temp.append(n)
		HiddenNodes.append(temp)
	# Make Output Layers
	for x in outputs:
		n = node(appFunc = x)
		n.addInputs(HiddenNodes[-1])
		OutputNodes.append(n)
	# Network created

	PrintNetwork()

	print('Network Constructed. Calculating result.')

	CalculateNN()
	# Will reach this point when result has been calculated and is within proper threshold results.
	print('\nWeights Found.')
	
	for x in StartingNodes:
		print(id(x), 'has starting value:', x.getValue())
	for y in HiddenNodes:
		for x in y:
			print(id(x), 'has hidden value:', x.getValue())
			print(id(x), 'has hidden error:', x.getError())
			print(id(x), 'had weights:', x.getWeights())
	for x in OutputNodes:
		print(id(x), 'has output value:', x.getValue())
		print(id(x), 'has output error:', x.getError())
		print(id(x), 'had weights:', x.getWeights())
	

def CalculateNN():
	global StartingNodes
	global HiddenNodes
	global OutputNodes
	global Threshold
	global AnswerSet
	global loops

	backprop = False

	# Forward propagation of solution
	print("\n", loops)
	for y in HiddenNodes:
		for x in y:
			x.calcValue()
			print('Hidden Value of', id(x), x.getValue(), 'with weights:', x.getWeights())
	for x in OutputNodes:
		x.calcValue()
		print('Output Value of', id(x), x.getValue(), 'with weights:', x.getWeights())
		x.calcOutputError(AnswerSet[OutputNodes.index(x)])
		print('Output Error of', id(x), x.getError())
	#	print('Correct to this point')
		if not ((x.getValue() <= (AnswerSet[OutputNodes.index(x)] + Threshold)) and (x.getValue() >= (AnswerSet[OutputNodes.index(x)] - Threshold))):
			backprop = True
	#PrintNetwork()
	if (loops < 15):
		loops += 1
		if (backprop == True):
	#		print('BackProping with', x.getError())
			BackPropNN()

def BackPropNN():
	global StartingNodes
	global HiddenNodes
	global OutputNodes
	global LearnRate

	for y in reversed(HiddenNodes):
		for x in y:
			x.calcHiddenError()
	#		print('H Error of', id(x), x.getError())

	#Need to make this parallel for multiple instances...

	for y in HiddenNodes:
		for x in y:
			x.updateHiddenWeights()
	#		print('H Value of', id(x), x.getValue(), 'with weights:', x.getWeights())
	for x in OutputNodes:
		x.updateOutputWeights()
	#	print('O Value of', id(x), x.getValue(), 'with weights:', x.getWeights())
	CalculateNN()

def PrintNetwork():
	global StartingNodes
	global HiddenNodes
	global OutputNodes
	print('\tStartingNodes:')
	for x in StartingNodes:
		#print(hex(id(x)), 'has inputs:', x.getInputs(), 'with weights:', x.getWeights())
		print(hex(id(x)), 'has value:', x.getValue(), 'with error:', x.getError())
		print(hex(id(x)), 'has outputs:', x.getOutputs(), 'with weight:', x.getWeightOutputs())
		print('')
	print('\tHiddenNodes:')
	for y in HiddenNodes:
		for x in y:
			print(hex(id(x)), 'has inputs:', x.getInputs(), 'with weights:', x.getWeights())
			print(hex(id(x)), 'has value:', x.getValue(), 'with error:', x.getError())
			print(hex(id(x)), 'has outputs:', x.getOutputs(), 'with weight:', x.getWeightOutputs())
			print('')
		print('\tnext level')
	print('\tOutputNodes:')
	for x in OutputNodes:
		print(hex(id(x)), 'has inputs:', x.getInputs(), 'with weights:', x.getWeights())
		print(hex(id(x)), 'has value:', x.getValue(), 'with error:', x.getError())
		print(hex(id(x)), 'has outputs:', x.getOutputs(), 'with weight:', x.getWeightOutputs())
		print('')



# This is a testing set. Build looks like:
	#
	#   2 - A - D 
	#     \   /   \
	#    	B       F - 101
	#     /   \   /
	#   3 - C - E
	#
if __name__== '__main__': main([2,3], [['S','S','S',], ['S', 'S']], ['L'], [101], threshold = 5, learnrate = 0.5)




