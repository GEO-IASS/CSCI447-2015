#!/usr/bin/python3

import sys
import random
import NN
#import Queue
from queue import *
import threading
import time
import logging
import rb_test

#USAGE: python3 handler_NN.py <infile> <outfile>  
                                                        
count = 0

# start a thread with the neural net calls for training and testing
def start_thread(inp, activation, out_activ, outp, learn, thresh, mmntm, logger):
    global count
    training_inputs = []
    training_data = []
    count += 1
    
    testNN = NN.main(inp, activation, out_activ, outp, learn, thresh, mmntm)
    print ("DONE TRAINING")
    for i in inp:
        for j in i:
            training_inputs.append(random.randint(0,4)) #create random inputs for testing
        training_data.append(training_inputs)
        training_inputs = []
    logger.info("ACTIVATION SET: ")
    logger.info(activation)
    logger.info("OUTPUT ACTIVATION: %s" % out_activ)
    logger.info("TESTING INPUT: ")
    logger.info(training_data)
    logger.info("OUTPUT: ")
    for x in training_data:
        testNN.SetStartingNodesValues(x)
        testNN.CalculateNNOutputs()
        logger.info(str(x))
        logger.info(testNN.GetNNResults())
        logger.info("RB OUTPUT: %s" % rb_test.rb_test(x))   
   
def setup_test(inputs, outputs, activation, out_activ):
    # is there anything we want to ask the user for as input?
    threshold       = 5
    learn_rate      = 0.3
    momentum        = 0.5
    out_activ       = []
    thread_count    = 0
    
    temp_input = inputs[(1*16):((1*16)+16)]
#        print (temp_input) # only want inputs/outputs in sets of 16 per dimension
    temp_out = outputs[(1*16):((1*16)+16)]
    print (temp_input)
    print (temp_out)
#    out_activ.append(activation[2][0])
    out_activ.append('S')                               
    logger = logging.getLogger('TEST-%s' % 5)
    logger.setLevel(logging.DEBUG)
    # file write handler
    file_handler = logging.FileHandler('3-5-Results.log')

    # custom formatter
    formatter = logging.Formatter('')
    file_handler.setFormatter(formatter)

    # register file handler
    logger.addHandler(file_handler)
    start_thread(temp_input, activation[4], out_activ, temp_out, learn_rate, 
                                                           threshold, momentum, logger)
#    for i in range(int(5)):
#        temp_input = inputs[(i*16):((i*16)+16)]
##        print (temp_input) # only want inputs/outputs in sets of 16 per dimension
#        temp_out = outputs[(i*16):((i*16)+16)]
##        print (temp_out)
#        for j in range(6): # there are 6 sets of activation test cases - 
#                           # see test_NN() for setup of activation
#            logger = logging.getLogger('thread-%s' % thread_count)
#            logger.setLevel(logging.DEBUG)
#
#            # file write handler
#            file_handler = logging.FileHandler('thread-%s.log' % thread_count)
#
#            # custom formatter
#            formatter = logging.Formatter('(%(threadName)-10s) %(message)s')
#            file_handler.setFormatter(formatter)
#
#            # register file handler
#            logger.addHandler(file_handler)
#
#            delay = random.random()
#            if j == 0:
#                # have to manually enter output activation (3rd arg) when 0 
#                # hidden layers
#                t = threading.Thread(target=start_thread, args = (temp_input,
#                        activation[j], ['S'], temp_out, learn_rate, threshold, 
#                                                              momentum,logger,))
#            elif j == 1:
#                # have to manually enter output activation (3rd arg) when 0 
#                # hidden layers
#                t = threading.Thread(target=start_thread, args = (temp_input,
#                        activation[j], ['S'], temp_out, learn_rate, threshold, 
#                                                         momentum, logger,))
#            else:
#                # add the first element in activation, either 'S' or 'L', 
#                # keeping all activation functions the same for all nodes
#                out_activ.append((activation[j][0][0]))                 
#                t = threading.Thread(target=start_thread, args = (temp_input,
#                    activation[j], out_activ, temp_out, learn_rate, threshold, 
#                                                           momentum, logger,))
#                out_activ = [] # reset to empty since only needs one element
#                t.start()
#            thread_count+=1
#
#    main_thread = threading.currentThread()
#    for t in threading.enumerate():
#        if t is not main_thread:
#            t.join()
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

