#include "softmax_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>

__inline__ __device__ float expsum(float* Z, int Z_y_dim, int row) {
    float psum = 0;
    for (int i = 0; i < Z_y_dim; i++) {
        psum += expf(Z[row * Z_y_dim + i]);
    }
    return psum;
}

// esum is row exponential sum
__inline__ __device__ float softmax(float Z_i, float esum) {
    return expf(Z_i) / esum;
}

__global__ void softmaxActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim * Z_y_dim) {
        int r = n / Z_y_dim;
        A[n] = softmax(Z[n], expsum(Z, Z_y_dim, r));
    }
}

__global__ void softmaxActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim * Z_y_dim) {
        int r = n / Z_y_dim;
        float smax = softmax(Z[n], expsum(Z, Z_y_dim, r));
        dZ[n] = dA[n] * smax * (1.f - smax);
    }
}

SoftmaxActivation::SoftmaxActivation(std::string name) {
    this->name = name;
}

SoftmaxActivation::~SoftmaxActivation() {}

Matrix& SoftmaxActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    softmaxActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax forward propagation.");

    return A;
}

Matrix& SoftmaxActivation::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    softmaxActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(), dZ.data_device.get(), Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax back propagation");

    return dZ;
}
