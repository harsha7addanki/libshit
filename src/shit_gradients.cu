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


// New backward kernels

__global__ void subtract_backward_kernel(
    const float* grad_out, float* grad_a, float* grad_b, int64_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&grad_a[i], grad_out[i]);
        atomicAdd(&grad_b[i], -grad_out[i]);
    }
}

__global__ void multiply_backward_kernel(
    const float* grad_out, const float* A, const float* B,
    float* grad_a, float* grad_b, int64_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&grad_a[i], grad_out[i] * B[i]);
        atomicAdd(&grad_b[i], grad_out[i] * A[i]);
    }
}

__global__ void negate_backward_kernel(
    const float* grad_out, float* grad_in, int64_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&grad_in[i], -grad_out[i]);
    }
}

__global__ void sigmoid_backward_kernel(
    const float* grad_out, const float* out, float* grad_in, int64_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        float s = out[i];
        atomicAdd(&grad_in[i], grad_out[i] * s * (1.0f - s));
    }
}

__global__ void tanh_backward_kernel(
    const float* grad_out, const float* out, float* grad_in, int64_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // tanh'(x) = 1 - tanh(x)^2
        float t = out[i];
        atomicAdd(&grad_in[i], grad_out[i] * (1.0f - t * t));
    }
}

__global__ void pow_backward_kernel(
    const float* grad_out, const float* A, float exponent,
    float* grad_a, int64_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // d/dx (x^e) = e * x^(e-1)
        atomicAdd(&grad_a[i], grad_out[i] * exponent * powf(A[i], exponent - 1.0f));
    }
}

__global__ void sum_all_backward_kernel(
    const float* grad_out, float* grad_in, int64_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&grad_in[i], grad_out[0]);
    }
}

__global__ void transpose_backward_kernel(
    const float* grad_out, float* grad_in, int64_t rows, int64_t cols)
{
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        // transpose is self-inverse: grad_in[y * cols + x] += grad_out[x * rows + y]
        atomicAdd(&grad_in[y * cols + x], grad_out[x * rows + y]);
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

    // New backward wrappers

    void backward_subtract(ShitTensor* grad_out, ShitTensor* A, ShitTensor* B, ShitTensor* grad_A, ShitTensor* grad_B) {
        int64_t size = A->size();
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        subtract_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), grad_A->gpu_ptr(), grad_B->gpu_ptr(), size);
        cudaDeviceSynchronize();
    }

    void backward_multiply(ShitTensor* grad_out, ShitTensor* A, ShitTensor* B, ShitTensor* grad_A, ShitTensor* grad_B) {
        int64_t size = A->size();
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        multiply_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), A->gpu_ptr(), B->gpu_ptr(),
            grad_A->gpu_ptr(), grad_B->gpu_ptr(), size);
        cudaDeviceSynchronize();
    }

    void backward_negate(ShitTensor* grad_out, ShitTensor* /*input*/, ShitTensor* grad_input) {
        int64_t size = grad_out->size();
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        negate_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), grad_input->gpu_ptr(), size);
        cudaDeviceSynchronize();
    }

    void backward_sigmoid(ShitTensor* grad_out, ShitTensor* out, ShitTensor* grad_input) {
        int64_t size = out->size();
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        sigmoid_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), out->gpu_ptr(), grad_input->gpu_ptr(), size);
        cudaDeviceSynchronize();
    }

    void backward_tanh(ShitTensor* grad_out, ShitTensor* out, ShitTensor* grad_input) {
        int64_t size = out->size();
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        tanh_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), out->gpu_ptr(), grad_input->gpu_ptr(), size);
        cudaDeviceSynchronize();
    }

    void backward_pow(ShitTensor* grad_out, ShitTensor* A, float exponent, ShitTensor* grad_A) {
        int64_t size = A->size();
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        pow_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), A->gpu_ptr(), exponent, grad_A->gpu_ptr(), size);
        cudaDeviceSynchronize();
    }

    void backward_sum_all(ShitTensor* grad_out, ShitTensor* input, ShitTensor* grad_input) {
        int64_t size = input->size();
        dim3 threads(256);
        dim3 blocks((size + 255) / 256);
        sum_all_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), grad_input->gpu_ptr(), size);
        cudaDeviceSynchronize();
    }

    void backward_transpose(ShitTensor* grad_out, ShitTensor* input, ShitTensor* grad_input) {
        auto shape = input->get_shape();
        int64_t rows = shape[0];
        int64_t cols = shape[1];
        dim3 threads(16, 16);
        dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
        transpose_backward_kernel<<<blocks, threads>>>(
            grad_out->gpu_ptr(), grad_input->gpu_ptr(), rows, cols);
        cudaDeviceSynchronize();
    }
}