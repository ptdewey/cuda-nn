#include "ce_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

#include <stdio.h>

#define MASK (unsigned int)0xffffffff

__inline__ __device__ float gClip(float i, float value) {
    if (isnan(i)) return value;
    return (fabsf(i) <= value) ? i : value;
}

__global__ void crossEntropyCost(float* predictions, float* target, 
                                 int N, int C, float* cost) {
    int t = threadIdx.x;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    int row = n / C;
    int c = n % C;
    float pc = (n < N * C) ? -1.f * (c == static_cast<int>(target[row])) *
        logf(predictions[n] + 1e-5f) : 0;

    pc += __shfl_down_sync(MASK, pc, 16);
    pc += __shfl_down_sync(MASK, pc, 8);
    pc += __shfl_down_sync(MASK, pc, 4);
    pc += __shfl_down_sync(MASK, pc, 2);
    pc += __shfl_down_sync(MASK, pc, 1);

    if (t % 32 == 0) {
        atomicAdd(cost, pc);
    }
}

__global__ void dCrossEntropyCost(float* predictions, float* target, 
                                  float* dY, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N * C) {
        int row = n / C;
        int c = n % C;
        // dY[n] = predictions[n] - (c == static_cast<int>(target[row]));
        dY[n] = gClip(predictions[n] - (c == static_cast<int>(target[row])), 1);
    }
}

CECost::CECost() {}

CECost::~CECost() {}

float CECost::cost(Matrix predictions, Matrix target) {
    assert(predictions.shape.x == target.shape.x);

    float cost = 0;
    float* d_cost;
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemset(d_cost, 0, sizeof(float));

    dim3 G(64);
    dim3 B = (predictions.shape.y * predictions.shape.x + G.x - 1) / G.x;

    crossEntropyCost <<< B, G >>> (predictions.data_device.get(), target.data_device.get(),
                                   predictions.shape.x, predictions.shape.y, d_cost);
    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute cross entropy cost.");

    cudaMemcpy(&cost, d_cost, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_cost);

    return cost;
}

Matrix CECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
    assert(predictions.shape.x == target.shape.x);

    dim3 G(64);
    dim3 B((predictions.shape.y * predictions.shape.x + G.x - 1) / G.x);

    dCrossEntropyCost <<< B, G >>> (predictions.data_device.get(), target.data_device.get(), 
                                    dY.data_device.get(), predictions.shape.x, predictions.shape.y);

    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for cross entropy.");

    return dY;
}
