#!/bin/bash

sep="_"
dot=".txt"
exe=$1
N="128 256 512"
T="20"
PROCNUM="1 4 8 16 32"
EDGE="1 3.14159265358979323846"
THR="4"

for n in $N; do
	for proc in $PROCNUM; do
		for e in $EDGE; do
			mpisubmit.pl -p $proc -t $THR --stdout "res_$proc$sep$n$sep$T$sep$e$dot" -- ./a.out $n $T $edge;
			while [ "$?" -ne "0" ]; do
				sleep 1;
				mpisubmit.pl -p $proc --stdout "res_$proc$sep$n$sep$T$sep$e$dot" -- ./a.out $n $T $edge;
			done
		done
	done
done
