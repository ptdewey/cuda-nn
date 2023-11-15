#include "bce_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

__global__ void binaryCrossEntropyCost(float *predictions, float *target,
                                       int N, float *cost) {

    int t = threadIdx.x;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // NOTE: block size is 256x1 for this kernel
    // PERF: bank size bottleneck is present for 1d block size
    __shared__ float s_pc[256];

    if (n < N) {
        float partial_cost = target[n] * logf(predictions[n]) + (1.0f - target[n]) * logf(1.0f - predictions[n]);

        // TODO: replace atomic with reduction
        // atomicAdd(cost, -partial_cost / N);

        s_pc[t] = (-1 * partial_cost) / N;

        if (t < 128) { s_pc[t] += s_pc[t + 128]; } __syncthreads();
        if (t < 64)  { s_pc[t] += s_pc[t + 64];  } __syncthreads();
        if (t < 32)  { s_pc[t] += s_pc[t + 32];  } __syncthreads();
        if (t < 16)  { s_pc[t] += s_pc[t + 16];  } __syncthreads();
        if (t < 8)   { s_pc[t] += s_pc[t + 8];   } __syncthreads();
        if (t < 4)   { s_pc[t] += s_pc[t + 4];   } __syncthreads();
        if (t < 2)   { s_pc[t] += s_pc[t + 2];   } __syncthreads();

        if (t < 1) {
            s_pc[t] += s_pc[t + 1];
            *cost = s_pc[t];
        }
    }
}

__global__ void dBinaryCrossEntropyCost(float *predictions, float *target,
                                        float *dY, int size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        // TODO: fix long memory reaches, writes
        dY[index] = -1.0 * (target[index] / predictions[index] -
            (1 - target[index]) / (1 - predictions[index]));
    }
}

float BCECost::cost(Matrix predictions, Matrix target) {
    assert(predictions.shape.x == target.shape.x);

    float *cost;
    // TODO: change to cudaMalloc maybe? (its just a single value so probably
    // fine)
    cudaMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    dim3 block_size(256);
    dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
    binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(
        predictions.data_device.get(), target.data_device.get(),
        predictions.shape.x, cost);
    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute binary cross entropy cost.");

    float cost_value = *cost;
    cudaFree(cost);

    return cost_value;
}

Matrix BCECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
    assert(predictions.shape.x == target.shape.x);

    dim3 block_size(256);
    dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
    dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(
        predictions.data_device.get(), target.data_device.get(),
        dY.data_device.get(), predictions.shape.x);
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for binary cross entropy.");

    return dY;
}
