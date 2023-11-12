all: build

build:
	nvcc -arch=sm_61 -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu

clean:
	rm main

run:
	./main

profile: 
	nvprof --csv --log-file profiler_out.csv --metrics all ./main


.PHONY: all build clean run
