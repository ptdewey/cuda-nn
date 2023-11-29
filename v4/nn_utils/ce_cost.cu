#include "ce_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

#define MASK (unsigned int)0xffffffff

__global__ void crossEntropyCost(float* predictions, float* target, 
                                 int N, int C, float* cost) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int row = n / C;
    int c = n % C;
    float y = (n < N * C) ? -1 * (c == target[row]) *
        logf(predictions[n] + 1e-5f) : 0;
    
    y += __shfl_down_sync(MASK, y, 16);
    y += __shfl_down_sync(MASK, y, 8);
    y += __shfl_down_sync(MASK, y, 4);
    y += __shfl_down_sync(MASK, y, 2);
    y += __shfl_down_sync(MASK, y, 1);

    if (threadIdx.x == 0) {
        atomicAdd(cost, y);
    }
}

__global__ void dCrossEntropyCost(float* predictions, float* target, 
                                  float* dY, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N * C) {
        int row = n / C;
        int c = n % C;
        dY[n] = -1 * (c == target[row]) / (predictions[n] + 1e-5f);
    }
}

CECost::CECost() {}

CECost::~CECost() {}

float CECost::cost(Matrix predictions, Matrix target) {
    assert(predictions.shape.x == target.shape.x);

    float* cost;
    float* d_cost;
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemset(d_cost, 0, sizeof(float));

    dim3 G(64);
    dim3 B = (predictions.shape.x + G.x - 1) / G.x;

    crossEntropyCost<<< B, G >>>(
        predictions.data_device.get(), target.data_device.get(),
        predictions.shape.x, predictions.shape.y, d_cost);
    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute cross entropy cost.");

    cudaMemcpy(cost, d_cost, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_cost);

    return *cost;
}

Matrix CECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
    assert(predictions.shape.x == target.shape.x);

    dim3 G(64);
    dim3 B((predictions.shape.x + G.x - 1) / G.x);
    dCrossEntropyCost<<< G, B >>>(
        predictions.data_device.get(), target.data_device.get(),
        dY.data_device.get(), predictions.shape.x, predictions.shape.y);
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for cross entropy.");

    return dY;
}
