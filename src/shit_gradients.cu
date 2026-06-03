#include <cuda_runtime.h>
#include "shit_gradients.hpp"
#include "shit_tensor.hpp"

__global__ void mse_backward_kernel(float* pred, float* target, float* grad_out, float* grad_pred, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float scale = grad_out[0];
        // Gradient of (p - t)^2 is 2 * (p - t), with external seed applied
        grad_pred[idx] = scale * (2.0f / (float)size) * (pred[idx] - target[idx]);
    }
}

__global__ void relu_backward_kernel(float* input, float* grad_output, float* grad_input, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0.0f;
    }
}

__global__ void add_backward_kernel(
    const float* grad_output,
    float* grad_x,
    float* grad_y,
    int64_t size) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&grad_x[i], grad_output[i]);
        atomicAdd(&grad_y[i], grad_output[i]);
    }
}

__global__ void mm_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ A,       
    const float* __restrict__ B,       
    float* __restrict__ grad_A,        
    float* __restrict__ grad_B,        
    int M, int K, int N) 
{
    // grad_A = grad_out * B^T
    // grad_B = A^T * grad_out

    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) { 
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += grad_out[row * N + i] * B[col * N + i];
        }
        atomicAdd(&grad_A[row * K + col], sum);
    }

    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[i * K + row] * grad_out[i * N + col];
        }
        atomicAdd(&grad_B[row * N + col], sum);
    }
}


namespace libshit::core {
    void backward_relu(ShitTensor* grad_out, ShitTensor* input, ShitTensor* grad_input) {
        int64_t size = input->size();
        
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        
        relu_backward_kernel<<<blocks, threads>>>(
            input->gpu_ptr(), grad_out->gpu_ptr(), grad_input->gpu_ptr(), size
        );

        cudaDeviceSynchronize();
    }

    void backward_matmul(ShitTensor* grad_out, ShitTensor* A, ShitTensor* B, ShitTensor* grad_A, ShitTensor* grad_B) {
        // all the sizing stuff for the kernel
        auto a_shape = A->get_shape();
        auto b_shape = B->get_shape();
        auto A_cols = a_shape[1];
        auto A_rows = a_shape[0];
        auto B_cols = b_shape[1];
        auto B_rows = b_shape[0];

        dim3 threads(16, 16);
        dim3 blocks((max(A_cols, B_cols) + 15) / 16, (max(A_rows, B_rows) + 15) / 16);
        
        // get all the cuda pointers from the tensors and launch
        mm_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), A->gpu_ptr(), B->gpu_ptr(), 
            grad_A->gpu_ptr(), grad_B->gpu_ptr(), 
            A_rows, A_cols, B_cols
        );

        // obviously we need to sync after launching the kernel
        cudaDeviceSynchronize();
    }


    void backward_add(ShitTensor* grad_out, ShitTensor* A, ShitTensor* B, ShitTensor* grad_A, ShitTensor* grad_B) {
        int64_t size = A->size();

        dim3 threads(256);
        dim3 blocks((size + 255) / 256);

        add_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), grad_A->gpu_ptr(), grad_B->gpu_ptr(), size
        );

        cudaDeviceSynchronize();
    }

    void backward_mse(ShitTensor* grad_out, ShitTensor* pred, ShitTensor* target, ShitTensor* grad_pred) {
        int64_t size = pred->size();
        
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        
        mse_backward_kernel<<<blocks, threads>>>(
            pred->gpu_ptr(), target->gpu_ptr(), grad_out->gpu_ptr(), grad_pred->gpu_ptr(), size
        );

        cudaDeviceSynchronize();
    }
}