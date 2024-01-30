#include "tanh_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>

__inline__ __device__ float _tanh(float z) {
    return (expf(z) - expf(-z)) / (expf(z) + expf(-z));
}

__global__ void tanhActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim * Z_y_dim) {
        A[n] = _tanh(Z[n]);
    }
}

__global__ void tanhActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim * Z_y_dim) {
        dZ[n] = dA[n] * (1 - _tanh(Z[n]) * _tanh(Z[n]));
    }
}

TanhActivation::TanhActivation(std::string name) {
    this->name = name;
}

TanhActivation::~TanhActivation() {}

Matrix& TanhActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);

    dim3 G(256);
    dim3 B((Z.shape.y * Z.shape.x + G.x - 1) / G.x);

    tanhActivationForward <<< B, G >>> (Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);

    NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax forward propagation.");

    return A;
}

Matrix& TanhActivation::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    dim3 G(256);
    dim3 B((Z.shape.y * Z.shape.x + G.x - 1) / G.x);

    tanhActivationBackprop <<< B, G >>> (Z.data_device.get(), dA.data_device.get(), dZ.data_device.get(), Z.shape.x, Z.shape.y);

    NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax back propagation");

    return dZ;
}
