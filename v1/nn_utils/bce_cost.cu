#include "bce_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

__global__ void binaryCrossEntropyCost(float *predictions, float *target,
                                       int N, float *cost) {

    int t = threadIdx.x;
    int n = blockIdx.x * blockIdx.x + threadIdx.x;

    __shared__ float s_pc[256];

    if (n < N) {
        float partial_cost = target[n] * logf(predictions[n]) + (1.0f - target[n]) * logf(1.0f - predictions[n]);

        // shared memory tree reduction
        s_pc[t] = (-1 * partial_cost) / N;

        if (t < 128) { s_pc[t] += s_pc[t + 128]; } __syncthreads();
        if (t < 64)  { s_pc[t] += s_pc[t + 64];  } __syncthreads();
        if (t < 32)  { s_pc[t] += s_pc[t + 32];  } __syncthreads();
        if (t < 16)  { s_pc[t] += s_pc[t + 16];  } __syncthreads();
        if (t < 8)   { s_pc[t] += s_pc[t + 8];   } __syncthreads();
        if (t < 4)   { s_pc[t] += s_pc[t + 4];   } __syncthreads();
        if (t < 2)   { s_pc[t] += s_pc[t + 2];   } __syncthreads();

        if (t == 0) {
            s_pc[t] += s_pc[t + 1];
            atomicAdd(cost, s_pc[t]);
        }
    }
}

__global__ void dBinaryCrossEntropyCost(float *predictions, float *target,
                                        float *dY, int size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        dY[index] = -1.0 * (target[index] / predictions[index] -
            (1 - target[index]) / (1 - predictions[index]));
    }
}

BCECost::BCECost() {}

BCECost::~BCECost() {}

float BCECost::cost(Matrix predictions, Matrix target) {
    assert(predictions.shape.x == target.shape.x);

    float* cost;
    float* d_cost;
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemset(d_cost, 0, sizeof(float));

    dim3 T(32, 32);
    int Bx = (predictions.shape.x + T.x - 1) / T.x;
    int By = (predictions.shape.y + T.y - 1) / T.y;
    dim3 B(Bx, By);
    binaryCrossEntropyCost<<< B, T >>>(
        predictions.data_device.get(), target.data_device.get(),
        predictions.shape.x, d_cost);
    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute binary cross entropy cost.");

    cudaMemcpy(cost, d_cost, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_cost);

    return *cost;
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
