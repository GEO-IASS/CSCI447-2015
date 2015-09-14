#!/bin/bash

FDATASETS=(ArtificialCharacters letter-recognition wine zoo) # 0 to 3
LDATASETS=(car cmc connect-4 ecoli tic-tac-toe yeast) # 0 to 5

echo ""
echo "Converting ARFFs to Nominal equivalents"
for i in `seq 0 3`;
do
	echo ${FDATASETS[$i]}.arff
	java -cp weka/ weka.filters.unsupervised.attribute.NumericToNominal -i ARFFs/${FDATASETS[$i]}.arff -o ARFFs/${FDATASETS[$i]}-alnom.arff -c first
	java -cp weka/ weka.filters.unsupervised.attribute.StringToNominal -R first-last -i ARFFs/${FDATASETS[$i]}-alnom.arff -o ARFFs/${FDATASETS[$i]}-nom.arff -c first
done
for i in `seq 0 5`;
do
	echo ${LDATASETS[$i]}.arff
	java -cp weka/ weka.filters.unsupervised.attribute.NumericToNominal -i ARFFs/${LDATASETS[$i]}.arff -o ARFFs/${LDATASETS[$i]}-alnom.arff -c last
	java -cp weka/ weka.filters.unsupervised.attribute.StringToNominal -R first-last -i ARFFs/${LDATASETS[$i]}-alnom.arff -o ARFFs/${LDATASETS[$i]}-nom.arff -c last
done

rm ARFFs/*alnom*

echo ""
echo "Making training and testing datasets"
for i in `seq 0 3`;
do
	echo ${FDATASETS[$i]}.arff
	java -cp weka/ weka.filters.unsupervised.instance.RemoveFolds -i ARFFs/${FDATASETS[$i]}-nom.arff -o ARFFs/${FDATASETS[$i]}-nom-train.arff -c first -N 5 -F 1 -V
	java -cp weka/ weka.filters.unsupervised.instance.RemoveFolds -i ARFFs/${FDATASETS[$i]}-nom.arff -o ARFFs/${FDATASETS[$i]}-nom-test.arff -c first -N 5 -F 1
done
for i in `seq 0 5`;
do
	echo ${LDATASETS[$i]}.arff
	java -cp weka/ weka.filters.unsupervised.instance.RemoveFolds -i ARFFs/${LDATASETS[$i]}-nom.arff -o ARFFs/${LDATASETS[$i]}-nom-train.arff -c last -N 5 -F 1 -V
	java -cp weka/ weka.filters.unsupervised.instance.RemoveFolds -i ARFFs/${LDATASETS[$i]}-nom.arff -o ARFFs/${LDATASETS[$i]}-nom-test.arff -c last -N 5 -F 1
done