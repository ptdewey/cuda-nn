#include "softmax_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>
#include <stdio.h>

// calculate the exponential sum of the elements of a (matrix)
__global__ void exp_sum(float* Z, float* Z_esum, int Z_x_dim, int Z_y_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /**
     * Z_y -> # of classes
     * Z_x -> batch size
     * row is an image, column is class
     */

    if (i < Z_x_dim) {
        float psum = 0;
        int row = i * Z_y_dim;
        for (int j = 0; j < Z_y_dim; j++) {
            // TODO: do more intelligent vector sum
            psum += expf(Z[row + j]); 
        }
        Z_esum[i] = psum;
    }
}

// Z_esum is row sum
__device__ float softmax(float Z_i, float esum) {
    return expf(Z_i) / (esum + 1e-10);
}


__global__ void softmaxActivationForward1(float* Z, float* Z_esum, float* A, int Z_x_dim, int Z_y_dim) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim) {
        // Compute the exponential sum for the current row of Z
        float exp_sum = 0.0f;
        for (int j = 0; j < Z_y_dim; ++j) {
            int index = n * Z_y_dim + j;
            exp_sum += expf(Z[index]);
        }

        // Store the exponential sum in Z_esum
        Z_esum[n] = exp_sum;

        // Compute the softmax activation for the current row of Z
        for (int j = 0; j < Z_y_dim; ++j) {
            int index = n * Z_y_dim + j;
            A[index] = expf(Z[index]) / exp_sum;
        }
    }
}

__global__ void softmaxActivationForward(float* Z, float* Z_esum, float* A, int Z_x_dim, int Z_y_dim) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim * Z_y_dim) {
        int y = n / (1.f * Z_y_dim);
        // printf("Zx: %d, n: %d, %lf, %d\n", Z_x_dim, n, n/(1.0*Z_y_dim), y);
        A[n] = softmax(Z[n], Z_esum[y]);
    }
}

__global__ void softmaxActivationBackprop1(float* Z, float* Z_esum, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim) {
        // Compute the gradient of the softmax activation for the current row of Z
        for (int j = 0; j < Z_y_dim; ++j) {
            int index = n * Z_y_dim + j;
            dZ[index] = dA[index] * (expf(Z[index]) / Z_esum[n] - expf(Z[index]) * expf(Z[index]) / (Z_esum[n] * Z_esum[n]));
        }
    }
}

__global__ void softmaxActivationBackprop(float* Z, float* Z_esum, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < Z_x_dim * Z_y_dim) {
        int y = n / Z_y_dim;
        float smax = softmax(Z[n], Z_esum[y]);
        dZ[n] = dA[n] * smax * (1 - smax);
    }
}

SoftmaxActivation::SoftmaxActivation(std::string name) {
    this->name = name;
}

SoftmaxActivation::~SoftmaxActivation()
{ }

Matrix& SoftmaxActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocateMemoryIfNotAllocated(Z.shape);

    // Z.x dimension is equal to batch size, which is usually small
    // - small thread block size is adapted to problem size, 64 might be better
    dim3 G(256);
    dim3 B((Z.shape.y * Z.shape.x + G.x - 1) / G.x);
    float* Z_esum;
    cudaMalloc(&Z_esum, Z.shape.x * sizeof(float));

    exp_sum<<< B, G >>>(Z.data_device.get(), Z_esum, Z.shape.x, Z.shape.y);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    softmaxActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), Z_esum, A.data_device.get(), Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax forward propagation.");

    return A;
}

Matrix& SoftmaxActivation::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(Z.shape);

    dim3 G(256);
    dim3 B((Z.shape.y * Z.shape.x + G.x - 1) / G.x);
    float* Z_esum;
    cudaMalloc(&Z_esum, Z.shape.x * sizeof(float));
    exp_sum<<< B, G >>>(Z.data_device.get(), Z_esum, Z.shape.x, Z.shape.y);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
    softmaxActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), Z_esum, dA.data_device.get(), dZ.data_device.get(), Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax back propagation");

    return dZ;
}
