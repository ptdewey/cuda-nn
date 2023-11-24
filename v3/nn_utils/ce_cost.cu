#include "ce_cost.hh"
#include "nn_exception.hh"

#include <assert.h>
#include <iostream>
#include <math.h>

// // convert 2D index into linear index
// __device__ int idx(int N, int nx, int ny) {
//     return ny * N + nx;
// }

#define MASK (unsigned int)0xffffffff

__global__ void crossEntropyCost(float* predictions, float* target, int N, int C, float* cost) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        float sampleCost = 0.0f;
        for (int j = 0; j < C; ++j) {
            // Indexing formula for accessing elements in 2D arrays
            int index = n * C + j;

            // One-hot encoded target for class j for the nth sample
            float y_ij = (j == target[n]) ? 1.0f : 0.0f;

            // Predicted probability for class j for the nth sample
            float y_hat_ij = predictions[index];

            // Add the cross-entropy term for class j
            sampleCost += y_ij * logf(y_hat_ij + 1e-10); // Small epsilon to avoid log(0)
        }

        // Accumulate the individual sample costs
        atomicAdd(cost, -sampleCost / C);

    }
}

__global__ void dCrossEntropyCost(float* predictions, float* target, float* dY, int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        // Compute gradients for the nth sample
        for (int j = 0; j < C; ++j) {
            // Indexing formula for accessing elements in 2D arrays
            int index = n * C + j;

            // One-hot encoded target for class j for the nth sample
            float y_ij = (j == target[n]) ? 1.0f : 0.0f;

            // Predicted probability for class j for the nth sample
            float y_hat_ij = predictions[index];

            // Compute the gradient for predicted probability y_hat_ij
            dY[index] = -y_ij / (y_hat_ij + 1e-10); // Small epsilon to avoid division by zero
        }
    }

    // if (n < N) {
    //     // TODO: more intelligent reduction
    //     for (int c = 0; c < C; c++) {
    //         int idx = n * C + c;
    //         dY[idx] = -1*target[n] + predictions[idx];
    //     }
    // }
}

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
