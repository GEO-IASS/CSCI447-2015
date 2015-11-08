#!/usr/bin/python3

import sys
import ES
import GA
import DE
import NN
import random

cRate = 0.5
mRate = 0.5
threshold = 5
generations = 1000
num_outs = 1
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
    
    resultsFile.write("DATASET: " + dataset + "\n")
    #resultsFile.write("ALGORITHM | Generations | Size | Participants | Victors | mRate | cRate | Threshold \n")
    #resultsFile.write("   " + str(algo) + "      |     " + str(generations) + "      |  " +
    #          str(size) + "  |     " + str(participants) + "       |    " + str(victors) + 
    #          "    |  " + str(mRate) + "  |  " + str(cRate) + "  |   " + str(threshold) + "     \n")

    dataIn = dataHandler()
    inputs = dataIn[0]
    outputs = dataIn[1]
    testInput = []
    testOutput = []
    learnrate = 0.3
    momentum = 0.5
    # Need 20% of inputs for testing
    for i in range((int(len(inputs)*0.8)+1), len(inputs)):
        x = random.choice(inputs)
        testInput.append(x)
        testOutput.append(outputs[inputs.index(x)])
        del outputs[inputs.index(x)]
        del inputs[inputs.index(x)]
    resultsFile.write("\nTest inputs: \n")
    for i in range(len(testInput)):
        resultsFile.write("%s " % testInput[i])
    resultsFile.write("\nTest expected outputs: \n")
    for i in range(len(testOutput)):
        resultsFile.write("%s " % testOutput[i])
    # Which algorithm gets chosen to run
    if algo in 'G':
        print("DOING GA TRAINING...")
        resultsFile.write("\nALGORITHM | Generations | Size | Participants | Victors | mRate | cRate | Threshold \n")
        resultsFile.write("   " + str(algo) + "      |     " + str(generations) + "      |  " + str(size) + "  |     " + str(participants) + "       |    " + str(victors) + "    |  " + str(mRate) + "  |  " + str(cRate) + "  |   " + str(threshold) + "     \n")
        testNN = GA.train(inputs, outputs, size, participants, victors, generations, threshold, cRate, mRate)
    elif algo in 'E':
        print("DOING ES TRAINING...")
        resultsFile.write("\nALGORITHM | Generations | Size | Participants | Victors | mRate | cRate | Threshold \n")
        resultsFile.write("   " + str(algo) + "      |     " + str(generations) + "      |  " + str(size) + "  |     " + str(participants) + "       |    " + str(victors) + "    |  " + str(mRate) + "  |  " + str(cRate) + "  |   " + str(threshold) + "     \n")
        testNN = ES.train(inputs, outputs, size, participants, victors, generations, threshold, cRate, mRate)
    elif algo in 'D':
        print("DOING DE TRAINING...")
        resultsFile.write("\nALGORITHM | Generations | Size | mRate | cRate | Threshold \n")
        resultsFile.write("   " + str(algo) + "      |     " + str(generations) + "      |  " +  str(size) + "    |  " + str(mRate) + "  |  " + str(cRate) + "  |   " + str(threshold) + "     \n")
        testNN = DE.train(inputs, outputs, size, generations, threshold, cRate, mRate)
    elif algo in 'B':
        print("DOING BP TRAINING...")
        resultsFile.write("\nALGORITHM | Generations | learnrate | momentum | Threshold \n")
        resultsFile.write("   " + str(algo) + "      |     " + str(generations) + "      |  " + str(learnrate) + "  |  " + str(momentum) + "  |   " + str(threshold) + "     \n")
        testNN = NN.main(inputs, [['S','S','S'], ['S','S']], ['S'], outputs, generations, learnrate, threshold, momentum)
    else:
        print("Unrecognized algorithm!")
        sys.exit()
    # Print test input/expected output - could be made prettier in a table
    # Start testing testNN
    for x in testInput:
        resultsFile.write("\nSet starting node vals\n")
        resultsFile.write("%s \n" % testNN.SetStartingNodesValues(x))
        testNN.CalculateNNOutputs()
        resultsFile.write("\nTest Input: " + str(x) + "\n")
        resultsFile.write("\nTest results: %s\n" % testNN.GetNNResults())
    resultsFile.write("\nRelative Error: {:2.2%} \n".format(NN.calcRelativeError(testNN, testInput, testOutput)))
    resultsFile.write("\nLeast Squares Error: %s \n" % NN.calcLeastSquaresError(testNN, testInput, testOutput))
    resultsFile.write("\nLoss Squared Error: %s \n" % NN.calcLossSquared(testNN, testInput, testOutput))
    resultsFile.write("\nPercent Misidentified: {:2.2%} \n".format(NN.calcPercentIncorrect(testNN, testInput, testOutput)))
    resultsFile.close()

# Set parameters via command line arguments
def evolve():
    global cRate, mRate, num_outs, threshold, generations, size, participants, victors, inFile, algo, dataset, resultsFile
    data = False
    alg = False

    for i in range(len(sys.argv)):
        if sys.argv[i] in '-h':
            printHelp()
            sys.exit()
        elif sys.argv[i] in '-m':
            mRate = float(sys.argv[i+1])
        elif sys.argv[i] in '-i':
            inFile = sys.argv[i+1]
        elif sys.argv[i] in '-t':
            threshold = float(sys.argv[i+1])
        elif sys.argv[i] in '-c':
            cRate = float(sys.argv[i+1])
        elif sys.argv[i] in '-g':
            generations = int(sys.argv[i+1])
        elif sys.argv[i] in '-s':
            size = int(sys.argv[i+1])
        elif sys.argv[i] in '-p':
            participants = int(sys.argv[i+1])
        elif sys.argv[i] in '-v':
            victors = int(sys.argv[i+1])
        elif sys.argv[i] in '-o':
            num_outs = int(sys.argv[i+1])
        elif sys.argv[i] in '-d':
            dataset = sys.argv[i+1]
            data = True
        elif sys.argv[i] in '-a':
            algo = sys.argv[i+1]
            alg = True
    if alg is False or data is False:
        print(alg, data)
        print("Need more information! Either algorithm or dataset name.")
        sys.exit()

    results = algo + "-" + dataset + ".txt"
    resultsFile = open(results,'w')
 
    if inFile in "":
        print("Need input file!")
        sys.exit()

def dataHandler():
    global inFile, num_outs
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
        for j in range(num_outs):
            temp.append(ins[i-(index+1)][0])
            del ins[i-(index+1)][0]
        outs.append(temp)
        temp = []
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
    print ("-o <num>         set number of outputs per input set")
    print ("-a <algo>        G(GA), E(ES), D(DE), or B(Backprop)")
    print ("-d <name>        add name for dataset for output file: algo-name.txt")
    print ("-i <dataFile>    enter the file for the dataset you want to test")
    print ("-h               print this help screen")
    print ("EXAMPLE CASE:")
    print ("handler.py -a G -c 0.5 -m 0.5 -t 5 -g 100000 -s 20 -v 5 -p 10 -i input.txt")

    # Parameters:
    # BP:   maxLoops=100000, learnrate=0.3, threshold=5, momentum=0.5
    # GA:   size=20, participants=10, victors=5, generations=100000, threshold=5, cRate=0.75, mRate=0.25
    # ES:   size=20, members=10, victors=5, generations=100000, threshold=5, cRate=0.75, mRate=0.25
    # DE:   size=20, threshold=5, generations=10000, cRate=0.4, mRate=0.6

if __name__ == '__main__': train_test()
