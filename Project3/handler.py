#!/usr/bin/python3

import sys
import ES
import GA
import DE
import NN

cRate = 0.5
mRate = 0.5
threshold = 10
generations = 100000
size = 20
participants = 10
victors = 5
inFile = ""


def trainNtest():
    global cRate, mRate, threshold, generations, size, participants, victors, inFile
    evolve()

    if algo in 'G':
        testNN = GA.train(inputs, outputs, size, participants, victors, 
                          generations, threshold, cRate, mRate)
    elif algo in 'E':
        testNN = ES.train(inputs, outputs, size, participants, victors, 
                          generations, threshold, cRate, mRate)
    elif algo in 'D':
        testNN = DE.train(inputs, outputs, size, participants, victors, 
                          generations, threshold, cRate, mRate)
    elif algo in 'B':
        testNN = NN.train(inputs, outputs, size, participants, victors, 
                          generations, threshold, cRate, mRate)
    else:
        print("Unrecognized algorithm!")
        sys.exit()
    
    calcRelativeError(testNN, inputs, outputs)
    calcLeastSqauresError(testNN, inputs, outputs)

def evolve():
    global cRate, mRate, threshold, generations, size, participants, victors, inFile

    for i in range(len(sys.argv)):
        if sys.argv[i] in '-h':
            printHelp()
            sys.exit()
        elif sys.argv[i] in '-m':
            mRate = sys.argv[i+1]
        elif sys.argv[i] in '-i':
            infile = sys.argv[i+1]
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
 
    if inFile in "":
        print("Need input file!")
        sys.exit()

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


