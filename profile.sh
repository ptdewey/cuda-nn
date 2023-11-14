#!/bin/bash


if [[ $# -eq 0 ]] ; then
    echo "Not enough arguments"
    exit 1
fi

cd ./$1
echo "Building in $1..."
nvcc -arch=sm_61 -DPROFILE=1 -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu
echo "Profiling in $1..."
nvprof --csv --log-file profiler_out_$1.csv --metrics all ./main
cd ..
echo "Done!"
