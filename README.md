# CUDA Neural Networks

## Usage
1. Clone the repository and cd into it
2. run `make build` to build the executable
3. run `./main {num epochs} {layer 1 size} {layer 2 size} {batch size} {num batches}` with your desired hyperparameters

- For binary classification (0s and 1s), there are 12000 total images, so batch size * num_batches must be less than or equal to 12000
- For multi-class classification (0s-9s), there are 60000 total images, so batch size * num_batches must be less than or equal to 60000
