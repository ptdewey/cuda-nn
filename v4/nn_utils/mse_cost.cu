#include "mse_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

#define MASK (unsigned int)0xffffffff

__global__ void meanSquareErrorCost(float* predictions, float* target, int N, int C, float* cost) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    int row = n / C;
    int c = n % C;
    float diff = (n < N * C) ? predictions[n] - (c == target[row]) : 0;
    float sum = (diff * diff) / C;

    // warp shuffle reduction
    sum += __shfl_down_sync(MASK, sum, 16);
    sum += __shfl_down_sync(MASK, sum, 8);
    sum += __shfl_down_sync(MASK, sum, 4);
    sum += __shfl_down_sync(MASK, sum, 2);
    sum += __shfl_down_sync(MASK, sum, 1);


    // thread blocks are usually pretty small so more reduction probably isn't necessary
    if (threadIdx.x == 0) {
        atomicAdd(cost, sum);
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
    // dim3 T(32, 32);
    // int Bx = (predictions.shape.x + T.x - 1) / T.x;
    // int By = (predictions.shape.y + T.y - 1) / T.y;
    // dim3 B(Bx, By);
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
    // dim3 block_size(32, 32);
    dim3 num_of_blocks((predictions.shape.y * predictions.shape.x + block_size.x - 1) / block_size.x);
    dMeanSquareErrorCost<<<num_of_blocks, block_size>>>(
        predictions.data_device.get(), target.data_device.get(),
        dY.data_device.get(), predictions.shape.x, predictions.shape.y);
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for mean square error.");

    return dY;
}
