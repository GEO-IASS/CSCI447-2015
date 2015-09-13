#!/bin/bash

FDATASETS=(ArtificialCharacters-test ArtificialCharacters-train letter-recognition zoo) # 0 to 4
LDATASETS=(car cmc connect-4 ecoli poker-test poker-train tic-tac-toe yeast) # 0 to 6
DATASETS=(ArtificialCharacters-test ArtificialCharacters-train letter-recognition zoo car cmc connect-4 ecoli poker-test poker-train tic-tac-toe yeast)

echo "Converting ARFFs to Nominal and Binary equivalents"
for i in `seq 0 3`;
do
	#echo ${FDATASETS[$i]}.arff
	(java -cp ":/Users/Coop/Github Repos/CSCI477-2015/project 1/weka.jar" weka.filters.unsupervised.attribute.NumericToNominal -i ARFFs/${FDATASETS[$i]}.arff -o ARFFs/${FDATASETS[$i]}-nom.arff -c first &)
	(java -cp ":/Users/Coop/Github Repos/CSCI477-2015/project 1/weka.jar" weka.filters.supervised.attribute.NominalToBinary -i ARFFs/${FDATASETS[$i]}.arff -o ARFFs/${FDATASETS[$i]}-bin.arff -c first &)
done
for i in `seq 0 7`;
do
	#echo ${LDATASETS[$i]}.arff
	(java -cp ":/Users/Coop/Github Repos/CSCI477-2015/project 1/weka.jar" weka.filters.unsupervised.attribute.NumericToNominal -i ARFFs/${LDATASETS[$i]}.arff -o ARFFs/${LDATASETS[$i]}-nom.arff -c last &)
	(java -cp ":/Users/Coop/Github Repos/CSCI477-2015/project 1/weka.jar" weka.filters.supervised.attribute.NominalToBinary -i ARFFs/${LDATASETS[$i]}.arff -o ARFFs/${LDATASETS[$i]}-bin.arff -c last &)
done

echo ""
for i in `seq 0 11`;
do
	(java -cp ":/Users/Coop/Github Repos/CSCI477-2015/project 1/weka.jar" weka.classifiers.lazy.IB1 -i ARFFs/${LDATASETS[$i]}.arff -o ARFFs/${LDATASETS[$i]}-bin.arff -c last &)
done