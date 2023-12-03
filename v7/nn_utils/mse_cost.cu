#include "mse_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

#define MASK (unsigned int)0xffffffff

__global__ void meanSquareErrorCost(float* predictions, float* target,
                                    int N, int C, float* cost) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

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


    // another reduction round isn't necessary due to small block size
    if (t % 32 == 0) {
        atomicAdd(cost, sum);
    }
}

__global__ void dMeanSquareErrorCost(float* predictions, float* target,
                                     float* dY, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N * C) {
        int row = n / C;
        int c = n % C;
        dY[n] = 2.f * (predictions[n] - (c == target[row])) / C;
    }
}

MSECost::MSECost() {}

MSECost::~MSECost() {}

float MSECost::cost(Matrix predictions, Matrix target) {
    assert(predictions.shape.x == target.shape.x);

    float cost = 0;
    float* d_cost;
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemset(d_cost, 0, sizeof(float));

    dim3 G(64);
    dim3 B((predictions.shape.y * predictions.shape.x + G.x - 1) / G.x);

    meanSquareErrorCost<<< B, G >>>(predictions.data_device.get(), target.data_device.get(),
                                    predictions.shape.x, predictions.shape.y, d_cost);
    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute MSE cost.");

    cudaMemcpy(&cost, d_cost, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_cost);

    return cost;
}

Matrix MSECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
    assert(predictions.shape.x == target.shape.x);

    dim3 G(64);
    dim3 B((predictions.shape.y * predictions.shape.x + G.x - 1) / G.x);

    dMeanSquareErrorCost <<< B, G >>> (predictions.data_device.get(), target.data_device.get(),
                                       dY.data_device.get(), predictions.shape.x, predictions.shape.y);

    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for mean square error.");

    return dY;
}
