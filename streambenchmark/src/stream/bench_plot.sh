#!/bin/bash -l

NX_MIN=10
NX_MAX=100000000
N_IT=20
N_IT_WARMUP=500

make

for BENCHMARK in base omp-host gpu
do
    #make clean
    #make stream-$BENCHMARK
    #echo 
    echo ---------------------------------
    echo stream-$BENCHMARK:
    echo ---------------------------------
    #printf "Buffersize GB/s\n" > ./$BENCHMARK-dat
    for (( i=NX_MIN; i<=NX_MAX; i*=10 ))
    do
        echo calculating for $BENCHMARK with buffersize $i
        ../../build/stream/stream-$BENCHMARK $i $N_IT_WARMUP $N_IT >> ./$BENCHMARK-dat
        echo
    done
done
echo
