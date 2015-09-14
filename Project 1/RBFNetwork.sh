#!/bin/bash

FDATASETS=(ArtificialCharacters letter-recognition wine zoo) # 0 to 3
LDATASETS=(car cmc connect-4 ecoli tic-tac-toe yeast) # 0 to 5

#Needs to be -R logistic or something
echo ""
echo "RBFNetwork"
for i in `seq 0 3`;
do
	(java -Xmx1024m -cp weka/ weka.classifiers.functions.RBFNetwork -i -c first -x 5 -k -t ARFFs/${FDATASETS[$i]}-nom-train.arff -T ARFFs/${FDATASETS[$i]}-nom-test.arff > Results/${FDATASETS[$i]}-RBFNetwork-results.out &)
	(java -Xmx1024m -cp weka/ weka.classifiers.functions.RBFNetwork -c first -x 10 -k -t ARFFs/${FDATASETS[$i]}-nom-train.arff -p first-last > Results/${FDATASETS[$i]}-RBFNetwork-predict.out &)
done
for i in `seq 0 5`;
do
	(java -Xmx1024m -cp weka/ weka.classifiers.functions.RBFNetwork -i -c last -x 5 -k -t ARFFs/${LDATASETS[$i]}-nom-train.arff -T ARFFs/${LDATASETS[$i]}-nom-test.arff > Results/${LDATASETS[$i]}-RBFNetwork-results.out &)
	(java -Xmx1024m -cp weka/ weka.classifiers.functions.RBFNetwork -c last -x 10 -k -t ARFFs/${LDATASETS[$i]}-nom-train.arff -p first-last > Results/${LDATASETS[$i]}-RBFNetwork-predict.out &)
done