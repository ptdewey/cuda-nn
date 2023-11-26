#!/bin/bash

dir=`pwd`

if [[ $# -eq 0 ]] ; then
    echo "Not enough arguments"
    exit 1
fi

if ! [[ -d $1 ]]; then
    echo "$1: No such file or directory"
    exit 1
fi

cd ./$1

echo "Building in $1..."
nvcc -arch=sm_61 -o main main.cu neural_network.cu mnist_dataset.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu

# this one takes a while, so it can be ignored by passing a param
# if [[ -z "$2" ]]; then 
#    echo "Profiling all metrics in $1..."
#    nvprof --csv --log-file $dir/profiler/profiler_metrics_out_$1.csv --metrics all ./main 1
# fi

echo "Profiling gpu-trace metrics in $1 with training size $2..."
nvprof --csv --log-file $dir/profiler/profiler_trace_out_$1.csv --print-gpu-trace ./main 0 512 128 2 $2

cd ..
echo "Done!"

