#!/usr/bin/python3

import sys
import ES
import GA
import DE
import NN
import random

cRate = 0.5
mRate = 0.5
threshold = 10
generations = 100000
size = 20
participants = 10
victors = 5
inFile = ""
algo = ""
dataset = ""
resultsFile = ""

def train_test():
    global cRate, mRate, threshold, generations, size, participants, victors, inFile, algo, dataset, resultsFile
    inputs = []
    outputs = []
    evolve()
    
    resultsFile.write("DATASET: " + dataset)
    resultsFile.write("ALGORITHM | Generations | Size | Participants | Victors | mRate | cRate | Threshold")
    resultsFile.write("   " + str(algo) + "      |     " + str(generations) + "      |  " +
              str(size) + "  |     " + str(participants) + "       |    " + str(victors) + 
              "    |  " + str(mRate) + "  |  " + str(cRate) + "  |   " + str(threshold) + "     ")

    dataIn = dataHandler()
    inputs = dataIn[0]
    outputs = dataIn[1]
    testInput = []
    testOutput = []
    learnrate = 0.5
    momentum = 0.5
    # Need 20% of inputs for testing
    for i in range((int(len(inputs)*0.8)+1), len(inputs)):
        testInput.append(random.choice(inputs))
        testOutput.append(outputs[inputs.index(
                                    testInput[i-((int(len(inputs)*0.8))+1)])])

    # Which algorithm gets chosen to run
    if algo in 'G':
        resultsFile.write(testNN = GA.train(inputs, outputs, size, participants, 
                         victors, generations, threshold, cRate, mRate))
    elif algo in 'E':
        resultsFile.write(testNN = ES.train(inputs, outputs, size, participants, 
                          victors, generations, threshold, cRate, mRate))
    elif algo in 'D':
        resultsFile.write(testNN = DE.train(inputs, outputs, size, participants, 
                          victors, generations, threshold, cRate, mRate))
    elif algo in 'B':
        resultsFile.write(testNN = NN.main(inputs, [['S','S','S'], ['S','S']], 
          ['S'], outputs, generations, learnrate, threshold, momentum))
    else:
        print("Unrecognized algorithm!")
        sys.exit()
    # Print test input/expected output - could be made prettier in a table
    resultsFile.write("Test inputs: ")
    restulsFile.write(testInput)
    resultsFile.write("Test expected outputs: ")
    resultsFile.write(testOutput)
    # Start testing testNN
    for x in testInput:
        resultsFile.write("Set starting node vals")
        resultsFile.write(testNN.SetStartingNodesValues(x))
        resultsFile.write("Calculate NN Outputs")
        resultsFile.write(testNN.CalculateNNOutputs())
        resultsFile.write("Test Input: " + str(x))
        resultsFile.write("Test results: " + testNN.GetNNResults())
    resultsFile.write("Relative error, least squares error:")
    resultsFile.write(calcRelativeError(testNN, inputs, outputs))
    resultsFile.write(calcLeastSqauresError(testNN, inputs, outputs))
    close(resultsFile)

def evolve():
    global cRate, mRate, threshold, generations, size, participants, victors, inFile, algo, dataset, resultsFile
    data = False
    alg = False

    for i in range(len(sys.argv)):
        if sys.argv[i] in '-h':
            printHelp()
            sys.exit()
        elif sys.argv[i] in '-m':
            mRate = sys.argv[i+1]
        elif sys.argv[i] in '-i':
            inFile = sys.argv[i+1]
        elif sys.argv[i] in '-t':
            threshold = sys.argv[i+1]
        elif sys.argv[i] in '-c':
            cRate = sys.argv[i+1]
        elif sys.argv[i] in '-g':
            generations = sys.argv[i+1]
        elif sys.argv[i] in '-s':
            size = sys.argv[i+1]
        elif sys.argv[i] in '-p':
            participants = sys.argv[i+1]
        elif sys.argv[i] in '-v':
            victors = sys.argv[i+1]
        elif sys.argv[i] in '-d':
            dataset = sys.argv[i+1]
            data = True
        elif sys.argv[i] in '-a':
            algo = sys.argv[i+1]
            alg = True
    if alg is False or data is False:
        print("Need more information! Either algorithm or dataset name.")
        sys.exit()

    results = algo + "-" + dataset + ".txt"
    resultsFile = open(results,'w')
 
    if inFile in "":
        print("Need input file!")
        sys.exit()

def dataHandler():
    global inFile
    index = 0

    with open(inFile, 'r') as f:
        content = f.readlines()
    for i in range(len(content)):
        if '@DATA' in content[i]:
            index = i
    if index == 0:
        print("Data needs @DATA section!")
        sys.exit()

    ins = []
    outs = [[]]
    temp = []
    for i in range(index+1, len(content)):
        ins.append(content[i].split())
        temp.append(ins[i-(index+1)][0])
        outs.append(temp)
        temp = []
        del ins[i-(index+1)][0]
    del outs[0]
    for i in range(len(outs)):
        outs[i][0] = int(outs[i][0])
        for j in range(len(ins[i])):
            ins[i][j] = float(ins[i][j])
    # need to parse input file for inputs and outputs    
    return ins, outs

def printHelp():
    print ("USAGE: handler.py [OPTIONS]")
    print ("OPTION           DESCRIPTION")
    print ("-c <num>         set the cRate to num")
    print ("-m <num>         set the mRate to num")
    print ("-t <num>         set the threshold to num")
    print ("-g <num>         set generations to num")
    print ("-v <num>         set victors to num")
    print ("-p <num>         set participants to num")
    print ("-a <algo>        G(GA), E(ES), D(DE), or B(Backprop)")
    print ("-h               print this help screen")
    print ("EXAMPLE CASE:")
    print ("handler.py -a G -c 0.5 -m 0.5 -t 5 -g 100000 -s 20 -v 5 -p 10 -i input.txt")

train_test()
