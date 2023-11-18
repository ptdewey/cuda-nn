all: build

build:
	nvcc -arch=sm_61 -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu

clean:
	rm main

run:
	./main

profile: 
	nvcc -arch=sm_61 -DPROFILE=1 -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu


.PHONY: all build clean run profile
