all: build

build:
	nvcc -arch=sm_61 -o main src/neural_network.cu src/mnist_dataset.cu src/coordinates_dataset.cu src/nn_utils/*.cu src/layers/*.cu src/main.cu 

clean:
	rm main

run:
	./main

.PHONY: all build clean run
