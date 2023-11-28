#include "mse_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

#define MASK (unsigned int)0xffffffff

__global__ void meanSquareErrorCost(float* predictions, float* target, int N, int C, float* cost) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    __shared__ float s_pc[32];

    int row = n / C;
    int c = n % C;
    float diff = (n < N * C) ? predictions[n] - (c == target[row]) : 0;
    float sum = (diff * diff) / C;

    s_pc[t] = sum;
    if (t < 16)  { s_pc[t] += s_pc[t + 16];  } __syncthreads();
    if (t < 8)   { s_pc[t] += s_pc[t + 8];   } __syncthreads();
    if (t < 4)   { s_pc[t] += s_pc[t + 4];   } __syncthreads();
    if (t < 2)   { s_pc[t] += s_pc[t + 2];   } __syncthreads();

    // update cost from final thread in block
    if (t == 0) {
        s_pc[t] += s_pc[t + 1];
        atomicAdd(cost, s_pc[t]);
    }
}

__global__ void dMeanSquareErrorCost(float* predictions, float* target, float* dY, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N * C) {
        int row = n / C;
        int c = n % C;
        // dY[n] = 2.0f * (predictions[n] - (c == static_cast<int>(target[row]))) / C;
        dY[n] = 2.0f * (predictions[n] - (c == target[row])) / C;
    }
}

MSECost::MSECost() {}

MSECost::~MSECost() {}

float MSECost::cost(Matrix predictions, Matrix target) {
    assert(predictions.shape.x == target.shape.x);

    float *cost;
    cudaMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    dim3 T(32);
    dim3 B((predictions.shape.y * predictions.shape.x + T.x - 1) / T.x);
    meanSquareErrorCost<<< B, T >>>(
        predictions.data_device.get(), target.data_device.get(),
        predictions.shape.x, predictions.shape.y, cost);
    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute MSE cost.");

    float cost_value = *cost;
    cudaFree(cost);

    return cost_value;
}

Matrix MSECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
    assert(predictions.shape.x == target.shape.x);

    dim3 block_size(64);
    dim3 num_of_blocks((predictions.shape.y * predictions.shape.x + block_size.x - 1) / block_size.x);
    dMeanSquareErrorCost<<<num_of_blocks, block_size>>>(
        predictions.data_device.get(), target.data_device.get(),
        dY.data_device.get(), predictions.shape.x, predictions.shape.y);
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for mean square error.");

    return dY;
}
