# USAGE: python rosenbrock.py <input>.txt <output>.txt runs total_sets
# where total_sets is the range of numbers to choose from for the values
# passed into rosenbrock, runs is the total times rosenbrock is computed 
# per dimension, must be less than 1/dimensions(total_sets)

#!/usr/bin/python
import random
import math
import sys

x_list = [] 
dimensions = []
infile = open(sys.argv[1], 'w')
outfile = open(sys.argv[2], 'w')
runs = int(sys.argv[3])  # num iterations for rb per dimension
total_sets = int(sys.argv[4])
all_outputs = []

def reset_x_list():
    global x_list
    for x in x_list:
        x_list.remove(x)
    for x in range(total_sets):
        x_list.append(x)

def make_x_list(num):
    global x_list
    global outfile
    global all_outputs

    not_done = 1
    i        = 1
    
    del dimensions[:]
    
    val = random.choice(x_list)
    dimensions.append(val)
#    x_list.remove(val)
    while (not_done):
        tmp = random.choice(x_list)
        while (tmp == val):
            tmp = random.choice(x_list)
        val = tmp
        if (i == num):
            not_done = 0
        else:
            dimensions.append(tmp)
#            x_list.remove(tmp)
            i+=1
    rosenbrock = 0 
    i = 0
    for i in range(len(dimensions) - 1):
        rosenbrock += (((1-dimensions[i])**2) + \
                       100*((dimensions[i+1] - \
                       (dimensions[i]**2))**2))
    s = str(rosenbrock)
    all_outputs.append(rosenbrock)
    outfile.write(s)
    outfile.write('\n')


def main():
    global runs
    global dimensions
    global infile
    global outfile

    reset_x_list()
    for i in range(5):
        for k in range(runs):
            make_x_list(i+2)
            for j in range(len(dimensions)):
                s = str(dimensions[j]) + " "
                infile.write(s)
            infile.write('\n')
            del dimensions[:]
    min_out = str(min(all_outputs))
    max_out = str(max(all_outputs))
    outfile.write(max_out)
    outfile.write(" ")
    outfile.write(min_out)
    outfile.write("\n")

#                for x in dimensions:
#                dimensions.remove(x)
#        reset_x_list()


if __name__=='__main__':main()
