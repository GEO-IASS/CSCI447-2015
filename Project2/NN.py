#!/usr/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	09/19/15
CSCI 447:	Project 2
"""

# Code for a neural network... Details coming to a theatre near you!!!

import sys
import math
import random

'''
help_screen = ["Usage   python NN.py <#input> <#hidden_layer> <#_output>"
               " <option> ...","OPTION  DESCRIPTION",
               "-r,-f   test either RBF(-r) or MLP(-f), must choose just one",
               "-i <#>  number of inputs", 
               "-h <#>  number of hidden layers",
               "-o <#>  number of outputs",
               "-g <#>  number of Gaussian basis functions",
               "-m      use momentum, default off", 
               "-s, -l  activation fn, sigmoid or linear"]

if sys.argv[1] in ['-h', '--h', '--help', '-help']:
    print ("\n".join(help_screen))
    sys.exit()
'''

global BiasWeight
BiasWeight = 0

class node:
	"""
	Code for a node in the neural network
	"""
	def __init__(self, functionType, name='NULL'):
		self.name = name
		self.edges = []
		self.value = random.random()
		self.functionType = functionType
	def getName(self):
		return self.name
	def addEdge(self, edge):
		#print('Added', edge.getName())
		self.edges.append(edge)
	def setValue(self):
		global BiasWeight
		for x in self.edges:
			#print(x.getBegin().getName())
			if (x.getBegin().isInput()):
				return x.getBegin().getValue()
			else:
				EdgeWeight = x.getWeight()
				NodeValue = x.getBegin().setValue()
				self.value += (EdgeWeight * NodeValue)
		# Comment out one of the following to implement that function:
		if self.functionType == 'L':
			self.value = [0,1][self.value > 0] # Step Function
		elif self.functionType == 'S':
			self.value = 1/(1+math.pow((math.e), (-self.value))) # Sigmoid Function
		return self.value
	def getValue(self):
		return self.value
	def isInput(self):
		return False

class edge:
	"""
	Code for the connecting edges between nodes
	"""
	def __init__(self, begin, end, name='NULL'):
		self.name = name
		self.begin = begin
		begin.addEdge(self)
		self.end = end
		end.addEdge(self)
		self.weight = random.random()
	def getName(self):
		return self.name
	def getBegin(self):
		return self.begin
	def getEnd(self):
		return self.end
	def updateWeight(self, weight):
		self.weight = weight
	def getWeight(self):
		return self.weight

class starting(node):
	"""
	Containers for the initial values
	"""
	def __init__(self, value, name='NULL'):
		self.name = name
		self.edges = []
		self.value = value
	def isInput(self):
		return True

def main():
	# This is a testing set. Build looks like:
	#
	#   A - 1 
	#     X   > 3 - OUT
	#   B - 2
	#
	A = starting(0, name = 'A')
	B = starting(0, name = 'B')
	node1 = node('S', name = '1')
	node2 = node('S', name = '2')
	node3 = node('S', name = '3')
	edgeA1 = edge(A, node1, name = 'A.1')
	edgeA2 = edge(A, node2, name = 'A.2')
	edgeB1 = edge(B, node1, name = 'B.1')
	edgeB2 = edge(B, node2, name = 'B.2')
	edge13 = edge(node1, node3, name = '1.3')
	edge23 = edge(node2, node3, name = '2.3')

	print(node3.setValue())
	# Currently no learning algorithm. Just computes inputs 
	# with randomized weights via step or sigmoid

def main2(inputs, layers, units, types):
	main()
	InputList = []
	NodesList = []
	y = 0
	for i in inputs: #List of starting nodes
		InputList.append(starting(i))
	for i in range(layers):
		for x in range(units[i]):
			if i == 0:
				#Construct grid of nodes
				NewNode = node(types[y])
				for j in InputList:
					edge(j, NewNode)
				NodesList.append(NewNode)
			else:
				NewNode = node(types[y])
				for j in NodesList[-x:]:
					edge(j, NewNode)
				NodesList.append(NewNode)
			y += 1
	for i in NodesList[-len(units[-1:]):]:
		print(i.setValue())



#if __name__ == '__main__': main()
if __name__== '__main__': main2([0,0], 2, [2,1], ['S','S','S'])


