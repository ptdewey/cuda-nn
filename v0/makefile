all: build

build:
	# nvcc -arch=sm_61 -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu
	nvcc -arch=sm_61 -o main neural_network.cu mnist_dataset.cu nn_utils/*.cu layers/*.cu main.cu 
	# nvcc -arch=sm_61 -o main neural_network.cu csv_dataset.cu nn_utils/*.cu layers/*.cu main.cu 

clean:
	rm main

run:
	./main

profile: 
	nvcc -arch=sm_61 -DPROFILE=1 -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu


.PHONY: all build clean run profile
