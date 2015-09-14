#!/bin/bash

FDATASETS=(ArtificialCharacters letter-recognition wine zoo) # 0 to 3
LDATASETS=(car cmc connect-4 ecoli tic-tac-toe yeast) # 0 to 5

echo ""
echo "J48"
for i in `seq 0 3`;
do
	(java -Xmx2048m -cp weka/ weka.classifiers.trees.J48 -U -i -c first -x 5 -k -t ARFFs/${FDATASETS[$i]}-nom-train.arff -T ARFFs/${FDATASETS[$i]}-nom-test.arff > Results/${FDATASETS[$i]}-J48-results.out &)
	(java -Xmx2048m -cp weka/ weka.classifiers.trees.J48 -c first -x 10 -k -t ARFFs/${FDATASETS[$i]}-nom-train.arff -p first-last > Results/${FDATASETS[$i]}-J48-predict.out &)
done
for i in `seq 0 5`;
do
	(java -Xmx2048m -cp weka/ weka.classifiers.trees.J48 -U -i -c last -x 5 -k -t ARFFs/${LDATASETS[$i]}-nom-train.arff -T ARFFs/${LDATASETS[$i]}-nom-test.arff > Results/${LDATASETS[$i]}-J48-results.out &)
	(java -Xmx2048m -cp weka/ weka.classifiers.trees.J48 -c last -x 10 -k -t ARFFs/${LDATASETS[$i]}-nom-train.arff -p first-last > Results/${LDATASETS[$i]}-J48-predict.out &)
done