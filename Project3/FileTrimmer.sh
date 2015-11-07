#/bin/bash

cd "/Users/Coop/GitHub Repos/CSCI447-2015/Project3/DataSets"

for y in `find . -type d \( ! -name . \) -mindepth 1 -maxdepth 1`; do
	echo ${y:2}
	echo "$(wc -l ${y:2}/data)"
	if [ "$(wc -l > $y/data)" -lt 300 ]
	then
		gshuf -n 300 $y/data > dataTrim
	else
		cat $y/data > dataTrim
	fi
done