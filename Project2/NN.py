#!/usr/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	09/19/15
CSCI 477:	Project 2
"""

# Code for a neural network... Details coming to a theatre near you!!!

import sys
import math
import random

global BiasWeight
BiasWeight = 0

class node:
	"""
	Code for a node in the neural network
	"""
	def __init__(self, name):
		self.name = name
		self.edges = []
		self.value = random.random()
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
		self.value = [0,1][self.value > 0] # Step Function
		#self.value = 1/(1+math.pow((math.e), (-self.value))) # Sigmoid Function
		return self.value
	def getValue(self):
		return self.value
	def isInput(self):
		return False

class edge:
	"""
	Code for the connecting edges between nodes
	"""
	def __init__(self, name, begin, end):
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
	def __init__(self, name, value):
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
	A = starting('A', 0)
	B = starting('B', 0)
	node1 = node('1')
	node2 = node('2')
	node3 = node('3')
	edgeA1 = edge('A.1', A, node1)
	edgeA2 = edge('A.2', A, node2)
	edgeB1 = edge('B.1', B, node1)
	edgeB2 = edge('B.2', B, node2)
	edge13 = edge('1.3', node1, node3)
	edge23 = edge('2.3', node2, node3)

	print(node3.setValue())
	# Currently no learning algorithm. Just computes inputs 
	# with randomized weights via step or sigmoid


if __name__ == '__main__': main()


