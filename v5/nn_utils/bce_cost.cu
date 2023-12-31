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

    if (n < N) {
        float pc = -1 * (target[n] * logf(predictions[n] + 1e-5f) + (1.f - target[n]) * logf(1.f - predictions[n] + 1e-5f)) / N;

        // shuffle reduction
        __syncwarp();
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

    float *cost;
    cudaMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    // dim3 block_size(256);
    dim3 T(32, 32);
    int Bx = (predictions.shape.x + T.x - 1) / T.x;
    int By = (predictions.shape.y + T.y - 1) / T.y;
    dim3 B(Bx, By);
    binaryCrossEntropyCost<<< B, T >>>(
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
    // dim3 block_size(32, 32);
    dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
    dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(
        predictions.data_device.get(), target.data_device.get(),
        dY.data_device.get(), predictions.shape.x);
    NNException::throwIfDeviceErrorsOccurred(
        "Cannot compute derivative for binary cross entropy.");

    return dY;
}
