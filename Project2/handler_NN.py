#!/usr/bin/python3

import sys
import random
import NN
#import Queue
from queue import *
import threading

#USAGE: python3 handler_NN.py <infile> <outfile>  
                                                        

# inputs - 2D array where len(input[i]) is the number of dimensions, 
# input[i][j] are the x values themselves for 2 - 6 dimensions
# inputs = set([])
# one output[i] per input[i]
# outputs = []

#def start_thread(q, inp, activation, out_activ, outp, learn, thresh, mmntm):
#    q.put(NN.main(inp, activation, out_activ, outp, learn, thresh, mmntm))

#q = Queue()

# start a thread with the neural net calls for training and testing
def start_thread(inp, activation, out_activ, outp, learn, thresh, mmntm):
    training_inputs = []
    training_data = []
    count = 0
    # scaling stuff
    maxim = 0
    for x in outp: maxim = max(maxim, max(x))
    minim = 10000
    for x in outp: minim = min(minim, min(x))
    
    testNN = NN.main(inp, activation, out_activ, outp, learn, thresh, mmntm)
    for i in inp:
        for j in i:
            training_inputs.append(random.randint(0,4)) #create random inputs for testing
        training_data.append(training_inputs)
        training_inputs = []
    for x in training_data:
        count += 1
        #print(x)
        testNN.SetStartingNodesValues(x)
        testNN.CalculateNNOutputs()
        outfile = open("out" + str(count) + ".txt", 'w')
        outfile.write(str(x))
        outfile.write(str(testNN.GetNNResults()))
        outfile.write('\n')
        outfile.close()
   
def setup_test(inputs, outputs, activation, out_activ):
    # is there anything we want to ask the user for as input?
    threshold       = 1
    learn_rate      = 0.5
    momentum        = 0.5
    out_activ       = []
    for i in range(int(5)):
        temp_input = inputs[(i*8):((i*8)+8)]
#        print (temp_input) # only want inputs/outputs in sets of 8 per dimension
        temp_out = outputs[(i*8):((i*8)+8)]
#        print (temp_out)
        for j in range(6): # there are 6 sets of activation test cases - 
                           # see test_NN() for setup of activation
#            print (out_activ)
            if j == 0:
                # have to manually enter output activation (3rd arg) when 0 
                # hidden layers
                t = threading.Thread(target=start_thread, args = (temp_input,
                        activation[j], ['S'], temp_out, learn_rate, threshold, 
                                                                    momentum,))
            elif j == 1:
                # have to manually enter output activation (3rd arg) when 0 
                # hidden layers
                t = threading.Thread(target=start_thread, args = (temp_input,
                        activation[j], ['S'], temp_out, learn_rate, threshold, 
                                                                    momentum,))
            else:
                # add the first element in activation, either 'S' or 'L', 
                # keeping all activation functions the same for all nodes
                out_activ.append((activation[j][0][0]))                 
                t = threading.Thread(target=start_thread, args = (temp_input,
                    activation[j], out_activ, temp_out, learn_rate, threshold, 
                                                                    momentum,))
                out_activ = [] # reset to empty since only needs one element
                t.start()

# This function pulls inputs and outputs from files and sets up the activations
def test_NN():
    inputs = [[]]
    outputs = []
    new_in = [[]]
    new_out = [[]]
    out_activation = [[] for x in range(1)]
    infile = sys.argv[1]
    outfile = sys.argv[2]
    num_tests = 5
    with open(infile) as textFile:
        inputs = [line.split() for line in textFile]
    new_in = [[int(string) for string in inner] for inner in inputs]

    #get activation functions for hidden layers - default 3 hidden
#    num_layers = int(sys.argv[3])
    num_units = len(new_in[0]) + 1
    activation = [[[]]for i in range(5)]# [num_layers][num_units]
    sigmoid = 'S'
    linear = 'L'
    
    activation = [[[]], [[]], [['S','S','S']], [['L', 'L','L']], [['S','S','S'],
                                        ['S','S']], [['L','L','L'],['L','L']]];

    out_activation.append([linear])
    out_activation.append([sigmoid])
#    print (activation)
    with open(outfile) as textFile:
        outputs = [line.split() for line in textFile]
    new_out = [[int(string) for string in inner] for inner in outputs]
#    print (new_in)
#    print ("\n")
#    print (new_out)
#    print (activation)
#    print(out_activation)
    setup_test(new_in, new_out, activation, out_activation)


test_NN()
