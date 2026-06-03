#include "libshit_core.h"
#include "../include/optim/optimizer.hpp"

__global__ void sgd_optimizer_kernel(float* weights, float* grad_weights, float learning_rate, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
}

namespace libshit::optim {
    void sgd(libshit::core::ShitTensor* weights, libshit::core::ShitTensor* grad_weights, float learning_rate) {
        int64_t size = weights->get_shape()[0] * weights->get_shape()[1];
        
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        
        sgd_optimizer_kernel<<<blocks, threads>>>(
            weights->gpu_ptr(), grad_weights->gpu_ptr(), learning_rate, size
        );

        cudaDeviceSynchronize();
    }
}
