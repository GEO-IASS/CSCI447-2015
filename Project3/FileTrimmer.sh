#/bin/bash

for y in `find DataSets/. -type d \( ! -name . \) -mindepth 1 -maxdepth 1`; do
	if [ "$(wc -l < $y/data)" -gt 300 ]
	then
		gshuf -n 300 $y/data > $y/dataTrim
	else
		cat $y/data > $y/dataTrim
	fi
done