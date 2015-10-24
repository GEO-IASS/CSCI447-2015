#!/usr/local/bin/python3

"""
Author: 	Clint Cooper, Emily Rohrbough, Leah Thompson
Date:   	10/23/15
CSCI 447:	Project 3
"""

import NN
import random
import copy

# Make base NN for template and fitness evaluations
baseNN = NN.NN([0, 0], [['S', 'S', 'S'], ['S', 'S']], ['S'], [0], learnrate=0.3, threshold=5, momentum=0.3)
baseNN.ConstructNetwork()

# Create citizens as arrays of weights that will be injected and stripped from the NN
citizenTemp = baseNN.GetNNWeights()

population = []
for i in range(10):
    for j in range(len(citizenTemp)):
        citizenTemp[j] = round(random.random(), 3)
    population.append(copy.deepcopy(citizenTemp))

for i in population: print(i)

# Bulk of GA code

    # Test each citizen and determine heroic level
    # If we have a hero go ahead and return him/her
    # Else, loop until a hero is found or we've reached max generations
        # Keep track of generation number starting at 0 and ended at our max generation count
        # Citizens are selected via random distribution and ranked based on heroics where we'll pull some number of potential heros
        # Have our heros mate (Crossover)
            # 2 parents where a random number of crossovers occur at random locations based on a passed in paramenter likelyhood
        # Have our heros children experience the world (Mutate)
            # random number of mutations that occur on random weights based on a passed in parameter of likelyhood
        # Test each child's heroic level
            # Determine if a child is a hero and if so, return child
        # Genocide of population is determined by elitism inverted on heroic level (cowardace).
        # Take number of children equal to number of tournament victors and reintroduce to the population
        # Increment Generation counter
    # return best hero we have if max generations is met.
