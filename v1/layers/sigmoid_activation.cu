#include "sigmoid_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>

__device__ float sigmoid(float x) {
    // TODO: replace with faster exp function
    // i.e. __expf or expf
    // if using __expf (fast math), compare accuracies
    // return 1.0f / (1 + __expf(-x));
    return 1.0f / (1 + expf(-x));
}

__global__ void sigmoidActivationForward(float* Z, float* A,
                                         int Z_x_dim, int Z_y_dim) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim * Z_y_dim) {
        // TODO: maybe use shared memory here?
        // might not be necessary, try other parts that actually require a reduction first
        A[n] = sigmoid(Z[n]);
    }
}

__global__ void sigmoidActivationBackprop(float* Z, float* dA, float* dZ,
                                          int Z_x_dim, int Z_y_dim) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim * Z_y_dim) {
        // TODO: shared memory storage for this part?
        // might not be necessary, try other parts that actually require a reduction first
        dZ[n] = dA[n] * sigmoid(Z[n]) * (1 - sigmoid(Z[n]));
    }
}

SigmoidActivation::SigmoidActivation(std::string name) {
    this->name = name;
}

SigmoidActivation::~SigmoidActivation()
{ }

Matrix& SigmoidActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    sigmoidActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
                                                            Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid forward propagation.");

    return A;
}

Matrix& SigmoidActivation::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
    sigmoidActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
                                                             dZ.data_device.get(),
                                                             Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid back propagation");

    return dZ;
}
