#!/usr/local/bin/python3

import sys
import PSO
import ACO
import kmeans
import dbscan
import compLearn

def cohesian(cluster):
    # Compare the intra distance of the cluster for all points and divide
    # by the number of instances in the cluster
    # We want to minimize this value
    coh = 0
    for p1 in cluster:
        for p2 in cluster:
            if p1 != p2:
                coh += PSO.EuclideanDistance(p1, p2)
    return coh / len(cluster)


def seperation(cluster1, cluster2):
    # calc the difference for every element in cluster1 to every element
    # in cluster2 and sum.
    # We want to maximize this value
    sep = 0
    for p1 in cluster1:
        for p2 in cluster2:
            sep += PSO.EuclideanDistance(p1, p2)
    return sep


def evalCluster(clusterSet):
    coh = 0
    sep = 0
    for c1 in clusterSet:
        coh += cohesian(c1)
        for c2i in range(clusterSet.index(c1), len(clusterSet)):
            sep += seperation(c1, clusterSet[c2i])
    return (coh, sep)


def rescaleSet(data, fromMinVal, fromMaxVal, toMinVal, toMaxVal):
    return [PSO.rescaleMatrix(x, fromMinVal, fromMaxVal, toMinVal, toMaxVal) for x in data]


def getInputData(fileName):
    index = 0
    with open(fileName, 'r') as f:
        content = f.readlines()
    for i in range(len(content)):
        if '@DATA' in content[i]:
            index = i
    if index == 0:
        print("Data needs @DATA section!")
        sys.exit()

    ins = []
    temp = []
    for i in range(index+1, len(content)):
        ins.append(content[i].split()[1:])
    # need to parse input file for inputs and outputs
    for i in range(len(ins)):
        for j in range(len(ins[i])):
            ins[i][j] = int(ins[i][j])
    return ins


def main(data):
    # Call one of the five algorithms with a set of input from source
    # Run cohesian and seperation calculations
    minVal = 10000
    maxVal = 0
    for x in data:
        for y in x:
            for z in y:
                minVal = min(minVal, z)
                maxVal = max(maxVal, z)
    inputs = rescaleSet(data, minVal, maxVal, 0, 1)  # Scale input data
    (coh, sep) = evalCluster(inputs)
    return (coh, sep)

if __name__ == '__main__':
    #data = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 127], [77, 76, 255], [38, 38, 127], [
    #    0, 0, 204], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]
    #clusters = [[[0, 0, 255], [0, 0, 0], [0, 0, 127], [77, 76, 255], [38, 38, 127], [0, 0, 204]],
    #            [[255, 0, 0], [255, 255, 255], [127, 0, 0], [255, 77, 76], [127, 38, 38], [204, 0, 0]],
    #            [[0, 255, 0], [0, 127, 0], [76, 255, 77], [38, 127, 38], [0, 204, 0]]]
    variables = (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    #            Type         DataSet      numParam     iterations   other       

    data = getInputData('DataSets/' + str(variables[1]))
    print("DataSet:", str(variables[1]))
    print("\nInputs:\n" + str(data))
    if variables[0] == 'K':
        clusters = kmeans.kmeans(data, int(variables[2]), int(variables[3]))
    elif variables[0] == 'D':
        clusters = dbscan.dbscan(data, len(data), len(data[0]) + 1)
    elif variables[0] == 'C':
        clusters = compLearn.compLearn(data, int(variables[2]), int(variables[3]), int(variables[4]))
    elif variables[0] == 'A':
        clusters = ACO.ACO(data, int(variables[2]), int(variables[3]))
    elif variables[0] == 'P':
        clusters = PSO.PSO(data, int(variables[2]), int(variables[3]))
    else:
        print('Unknown function specified...')
        sys.exit(1)
    print("\nClusters:")
    for x in clusters:
        print(x)
    (coh, sep) = main(clusters)
    print("\nNumClusters:", len(clusters))
    print("\nNumPerCluster:", [len(x) for x in clusters])
    print("\nCohesion:", coh)
    print("\nSeperation: ", sep)


    #data = getInputData('DataSets/blood.txt')
    #print("Inputs:\n" + str(data))
    ##clusters = PSO.PSO(data, 5, 1000)
    ##clusters = ACO.ACO(data, 5, 10000)
    ##clusters = kmeans.kmeans(data, 5, 10000)
    ##clusters = dbscan.dbscan(data, len(data), len(data[0]) + 1) # eps = 150?, minPts = len(data[0]) + 1
    ##clusters = compLearn.compLearn(data, 5, 10000, 0.005) 
    ##print("\nClusters:\n" + str(clusters))
    #(coh, sep) = main(clusters)
    #print("\nNumClusters:", len(clusters))
    #print("\nNumber of points per cluster:", [len(x) for x in clusters])
    #print("\nCohesian:   ", coh)
    #print("\nSeperation: ", sep)
