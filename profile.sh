#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "Not enough arguments"
    exit 1
fi

if ! [[ -d $1 ]]; then
    echo "$1: No such file or directory"
    exit 1
fi

cd ./$1

if ! [[ -f ./main ]]; then
    echo "Building in $1..."
    nvcc -arch=sm_61 -DPROFILE=1 -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu
fi

# this one takes a while, so it can be ignored by passing a param
if [[ -z "$2" ]]; then 
    echo "Profiling all metrics in $1..."
    nvprof --csv --log-file profiler_out_$1.csv --metrics all ./main
fi

echo "Profiling gpu-trace metrics in $1..."
nvprof --csv --log-file profiler_trace_out_$1.csv --print-gpu-trace --print-api-trace ./main
# nvprof --csv --log-file profiler_trace_out_$1.csv --print-gpu-trace ./main

cd ..
echo "Done!"

