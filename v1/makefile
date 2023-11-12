# Makefile

# Default target
all: build

# Build target
build:
	nvcc -o main main.cu neural_network.cu coordinates_dataset.cu nn_utils/*.cu layers/*.cu

# Clean target
clean:
	rm main

# Run target
run:
	./main

.PHONY: all build clean run
