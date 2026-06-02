#include <cuda_runtime_api.h>
#include <cmath>
#include "../include/shit_tensor.hpp"
#include "../include/shit_errors.hpp"
#include "../include/shit_gradients.hpp"

__global__ void mse_kernel(float* target, float* pred, float* total, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float diff = target[idx] - pred[idx];
        float squared_diff = diff * diff;
        atomicAdd(total, squared_diff);
    }
}

__global__ void relu_kernel(float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void mm_kernel(float* input_a, float* input_b, float* output, int64_t a_rows, int64_t a_cols, int64_t b_cols) {
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < b_cols && y < a_rows) {
        // https://en.wikipedia.org/wiki/Matrix_multiplication#Definitions
        float sum = 0.0f;
        for (int64_t k = 0; k < a_cols; ++k) {
            sum += input_a[y * a_cols + k] * input_b[k * b_cols + x];
        }
        output[y * b_cols + x] = sum;
    }
}

__global__ void add_kernel(float* input_a, float* input_b, float* output, int64_t a_rows, int64_t a_cols, int64_t b_cols) {
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < b_cols && y < a_rows) {
        output[y * b_cols + x] = input_a[y * a_cols + x] + input_b[y * b_cols + x];
    }
}

namespace libshit::core{

    SHIT_API ShitTensor* relu(ShitTensor* tensor) {
        int threads_per_block = 256;
        int64_t blocks = (tensor->size() + threads_per_block - 1) / threads_per_block;
        ShitTensor* out_tensor = new ShitTensor(tensor->tensor_shape());

        if (ShitTape::instance().is_active()) {
            // We create a node that knows how to link these tensors
            auto node = std::make_unique<ReLUNode>(tensor, out_tensor);
            ShitTape::instance().push_node(std::move(node));
        }
    
        // kernel launch
        relu_kernel<<<blocks, threads_per_block>>>(tensor->gpu_ptr(), out_tensor->gpu_ptr(), tensor->size());
    
        // make sure that no error happens
        SHIT_CHECK(cudaGetLastError());
        SHIT_CHECK(cudaDeviceSynchronize());
        return out_tensor;
    }
    
    SHIT_API ShitTensor* matmul(ShitTensor* tensora, ShitTensor* tensorb) {
        const std::vector<int64_t>& a_shape = tensora->get_shape();
        const std::vector<int64_t>& b_shape = tensorb->get_shape();
        
        if (a_shape.size() < 2 || b_shape.size() < 2 || a_shape[1] != b_shape[0]) {
            std::cerr << "LIBSHIT MATMUL ERROR: Inner matrix dimensions mismatch! " 
                      << "Cannot multiply (" << a_shape[0] << "x" << a_shape[1] << ") by ("
                      << b_shape[0] << "x" << b_shape[1] << ")" << std::endl;
            exit(1);
        }
        int a_rows = a_shape[0];
        int a_cols = a_shape[1];
        int b_cols = b_shape[1];
        
        ShitTensor* out_tensor = new ShitTensor({a_rows, b_cols});
        
        if (ShitTape::instance().is_active()) {
            // We create a node that knows how to link these tensors
            auto node = std::make_unique<MatMulNode>(tensora, tensorb, out_tensor);
            ShitTape::instance().push_node(std::move(node));
        }
    
        // same thing as 256 but in 2d vector 16*16 = 256
        dim3 threads_per_block(16, 16);
    
        dim3 blocks_per_grid(
            (b_cols + threads_per_block.x - 1) / threads_per_block.x,
            (a_rows + threads_per_block.y - 1) / threads_per_block.y
        );
    
        // kernel launch
        mm_kernel<<<blocks_per_grid, threads_per_block>>>(tensora->gpu_ptr(), tensorb->gpu_ptr(), out_tensor->gpu_ptr(), a_rows, a_cols, b_cols);
    
        // make sure that no error happens
        SHIT_CHECK(cudaGetLastError());
        SHIT_CHECK(cudaDeviceSynchronize());
        return out_tensor;
    }

    SHIT_API ShitTensor* add(ShitTensor* tensora, ShitTensor* tensorb) {
        const std::vector<int64_t>& a_shape = tensora->get_shape();
        const std::vector<int64_t>& b_shape = tensorb->get_shape();
        
        if (a_shape != b_shape) {
            std::cerr << "LIBSHIT ADD ERROR: Inner matrix dimensions mismatch! " 
                      << "Cannot add (" << a_shape[0] << "x" << a_shape[1] << ") and ("
                      << b_shape[0] << "x" << b_shape[1] << ")" << std::endl;
            exit(1);
        }
        int a_rows = a_shape[0];
        int a_cols = a_shape[1];
        int b_cols = b_shape[1];
        
        ShitTensor* out_tensor = new ShitTensor({a_rows, b_cols});
        
        if (ShitTape::instance().is_active()) {
            // We create a node that knows how to link these tensors
            auto node = std::make_unique<AddNode>(tensora, tensorb, out_tensor);
            ShitTape::instance().push_node(std::move(node));
        }
    
        // same thing as 256 but in 2d vector 16*16 = 256
        dim3 threads_per_block(16, 16);
    
        dim3 blocks_per_grid(
            (b_cols + threads_per_block.x - 1) / threads_per_block.x,
            (a_rows + threads_per_block.y - 1) / threads_per_block.y
        );
    
        // kernel launch
        add_kernel<<<blocks_per_grid, threads_per_block>>>(tensora->gpu_ptr(), tensorb->gpu_ptr(), out_tensor->gpu_ptr(), a_rows, a_cols, b_cols);
    
        // make sure that no error happens
        SHIT_CHECK(cudaGetLastError());
        SHIT_CHECK(cudaDeviceSynchronize());
        return out_tensor;
    }

    SHIT_API ShitTensor* mse(ShitTensor* pred, ShitTensor* target) {
        int threads_per_block = 256;
        int64_t size = pred->size();
        ShitTensor* out_tensor = new ShitTensor({1});

        if (ShitTape::instance().is_active()) {
            auto node = std::make_unique<MSENode>(pred, target, out_tensor);
            ShitTape::instance().push_node(std::move(node));
        }
        auto blocks = (size + 255) / 256;
        mse_kernel<<<blocks, threads_per_block>>>(target->gpu_ptr(), pred->gpu_ptr(), out_tensor->gpu_ptr(), size);

        SHIT_CHECK(cudaGetLastError());
        SHIT_CHECK(cudaDeviceSynchronize());
        return out_tensor;
    }
}