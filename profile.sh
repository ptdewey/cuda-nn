#!/bin/bash


if [[ $# -eq 0 ]] ; then
    echo "Not enough arguments"
    exit 1
fi

cd ./$1
echo "Building in $1..."
nvcc -arch=sm_61 -DPROFILE=1 -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu

echo "Profiling all metrics in $1..."
nvprof --csv --log-file profiler_out_$1.csv --metrics all ./main

echo "Profiling gpu-trace metrics in $1..."
nvprof --csv --log-file profiler_trace_out_$1.csv --print-gpu-trace --print-api-trace ./main
# nvprof --csv --log-file profiler_trace_out_$1.csv --print-gpu-trace ./main

cd ..
echo "Done!"
