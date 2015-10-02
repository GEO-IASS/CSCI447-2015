#!/usr/bin/python3i

import sys
import random
import NN

# inputs - 2D array where len(input[i]) is the number of dimensions, 
# input[i][j] are the x values themselves for 2 - 6 dimensions
# inputs = set([])
# one output[i] per input[i]
# outputs = []

def setup_NN(inPut, outPut):
    # is there anything we want to ask the user for as input?
    index = 0
    activation_func = 1 # 1 is sigmoid, 0 is logistic
    threshold = 0.001
    learn_rate = 0.5
    momentum = 0.2
    for i in inPut:
        # hidden is a set , where each element is # of levels for that layer 
        # followed by the activation functions per level
        # hidden[j] = [5, 1, 1, 1, 1, 1] - 5 levels with sigmoid applied to each
        hidden = set([])
        num_layers = random.randint(1, 5)
        for j in range(num_layers):
            hidden[j].append(random.randint(1, 5))
            for k in range(hidden[j][0])
                hidden[j].append(activation_func)
        # call NN.py main method to run the NN
        NN.main(i, outPut[index], num_layers, hidden, [0.2], threshold, 
                learn_rate, momentum)
        index+=1


def test_NN():
    inputs = set([])
    outputs = []
    infile = sys.argv[1]
    outfile = sys.argv[2]
    inputs = map(str.split, open(infile))
    outputs = [line.rstrip('\n') for line in open(outfile)]
    
    setup_NN(inputs, outputs)


test_NN()
