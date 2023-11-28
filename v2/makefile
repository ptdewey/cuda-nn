all: build

build:
	nvcc -arch=sm_61 -o main main.cu neural_network.cu mnist_dataset.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu

clean:
	rm main

run:
	./main

coord:
	nvcc -arch=sm_61 -o main main.cu neural_network.cu mnist_dataset.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu -DTEST=0 -DCOORD=0

.PHONY: all build clean run coord
