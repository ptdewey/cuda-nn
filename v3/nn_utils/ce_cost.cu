#include "ce_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

#define MASK (unsigned int)0xffffffff

__global__ void crossEntropyCost(float* predictions, float* target, int N, int C, float* cost) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        float sampleCost = 0.0f;
        for (int j = 0; j < C; ++j) {
            int index = n * C + j;
            float y_ij = (j == target[n]) ? 1.0f : 0.0f;
            float y_hat_ij = predictions[index];
            sampleCost += y_ij * logf(y_hat_ij + 1e-5);
        }
        // TODO: more intelligent reduction
        // TEST: /C ? should maybe be N??
        atomicAdd(cost, -sampleCost / C);
    }
}

__global__ void dCrossEntropyCost(float* predictions, float* target, float* dY, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        for (int j = 0; j < C; ++j) {
            int index = n * C + j;
            float y_ij = (j == target[n]) ? 1.0f : 0.0f;
            float y_hat_ij = predictions[index];
            dY[index] = -y_ij / (y_hat_ij + 1e-5);
        }
    }
}

CECost::CECost() {}

CECost::~CECost() {}

float CECost::cost(Matrix predictions, Matrix target) {
    assert(predictions.shape.x == target.shape.x);

    float *cost;
    cudaMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    // TODO: check block sizes
    dim3 G(256);
    dim3 B = (predictions.shape.x + G.x - 1) / G.x;
    // dim3 G(32, 32);
    // int Bx = (predictions.shape.x + G.x - 1) / G.x;
    // int By = (predictions.shape.y + G.y - 1) / G.y;
    // dim3 B(Bx, By);

    crossEntropyCost<<< B, G >>>(
        predictions.data_device.get(), target.data_device.get(),
        predictions.shape.x, predictions.shape.y, cost);
    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute cross entropy cost.");

    float cost_value = *cost;
    cudaFree(cost);

    return cost_value; // / predictions.shape.y;
}

Matrix CECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
    assert(predictions.shape.x == target.shape.x);

    dim3 G(256);
    // dim3 G(32, 32);
    dim3 B((predictions.shape.x + G.x - 1) / G.x);
    // int Bx = (predictions.shape.x + G.x - 1) / G.x;
    // int By = (predictions.shape.y + G.y - 1) / G.y;
    // dim3 B(Bx, By);
    dCrossEntropyCost<<< G, B >>>(
        predictions.data_device.get(), target.data_device.get(),
        dY.data_device.get(), predictions.shape.x, predictions.shape.y);
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for cross entropy.");

    return dY;
}
