# CUDA Neural Networks

## Usage
1. Clone the repository and cd into it
2. run `make build DIR=v0` to build the v0 version (change 0 to desired version)
3. cd into version directory and run `./main {num epochs} {layer 1 size} {layer 2 size} {batch size} {num batches}` with your desired hyperparameters

- For v0 through v2.5, there are 12000 total images, so batch size * num batches must be less than or equal to 12000
- For v3 through v6, there are 60000 total images, so batch size * num batches must be less than or equal to 60000
