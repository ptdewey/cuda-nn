#include "bce_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

// convert 2D index into linear index
__device__ int idx(int N, int nx, int ny) {
    return ny * N + nx;
}

#define MASK (unsigned int)0xffffffff

__global__ void binaryCrossEntropyCost(float *predictions, float *target,
                                       int N, float *cost) {
    // switched to 2D thread blocks
    int t = threadIdx.x;
    int w = threadIdx.y;
    int nx = blockDim.x * blockIdx.x + threadIdx.x;
    int ny = blockDim.y * blockIdx.y + threadIdx.y;
    int n = idx(N, nx, ny);

    __shared__ float w_pc[32];

    float pc = (n < N) ? -1 * (target[n] * logf(predictions[n] + 1e-5f) + 
        (1.f - target[n]) * logf(1.f - predictions[n] + 1e-5f)) / N : 0;

    // shuffle reduction
    pc += __shfl_down_sync(MASK, pc, 16);
    pc += __shfl_down_sync(MASK, pc, 8);
    pc += __shfl_down_sync(MASK, pc, 4);
    pc += __shfl_down_sync(MASK, pc, 2);
    pc += __shfl_down_sync(MASK, pc, 1);


    // apppend results from first thread in each warp
    if (t == 0) {
        w_pc[w] = pc;
    } 

    __syncthreads();

    // reduce full results in warp 0
    if (w == 0) {
        pc = w_pc[t];

        pc += __shfl_down_sync(MASK, pc, 16);
        pc += __shfl_down_sync(MASK, pc, 8);
        pc += __shfl_down_sync(MASK, pc, 4);
        pc += __shfl_down_sync(MASK, pc, 2);
        pc += __shfl_down_sync(MASK, pc, 1);

        // thread 0 in warp 0 adds to cost accumulator
        if (t == 0) {
            atomicAdd(cost, pc);
        }
    }
}

__global__ void dBinaryCrossEntropyCost(float *predictions, float *target,
                                        float *dY, int size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        dY[index] = -1 * (target[index] / (predictions[index] + 1e-5f) -
            (1 - target[index]) / (1 - predictions[index] + 1e-5f));
    }
}

BCECost::BCECost() {}

BCECost::~BCECost() {}

float BCECost::cost(Matrix predictions, Matrix target) {
    assert(predictions.shape.x == target.shape.x);

    float cost = 0;
    float* d_cost;
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemset(d_cost, 0, sizeof(float));

    dim3 G(32, 32);
    int Bx = (predictions.shape.x + G.x - 1) / G.x;
    int By = (predictions.shape.y + G.y - 1) / G.y;
    dim3 B(Bx, By);

    binaryCrossEntropyCost <<< B, G >>> (predictions.data_device.get(), target.data_device.get(), 
                                         predictions.shape.x, d_cost);

    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute binary cross entropy cost.");

    cudaMemcpy(&cost, d_cost, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_cost);

    return cost;
}

Matrix BCECost::dCost(Matrix predictions, Matrix target, Matrix dY) {
    assert(predictions.shape.x == target.shape.x);

    dim3 G(64);
    dim3 B((predictions.shape.x + G.x - 1) / G.x);

    dBinaryCrossEntropyCost <<< B, G >>> (predictions.data_device.get(), target.data_device.get(),
                                          dY.data_device.get(), predictions.shape.x);

    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for binary cross entropy.");

    return dY;
}
