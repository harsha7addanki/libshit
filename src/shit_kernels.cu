#include <cuda_runtime.h>
#include <cmath>
#include "shit_tensor.hpp"
#include "shit_errors.hpp"
#include "shit_gradients.hpp"

// ========== Forward Kernels ==========

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

// New forward kernels

__global__ void subtract_kernel(float* a, float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void multiply_kernel(float* a, float* b, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void negate_kernel(float* input, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = -input[idx];
    }
}

__global__ void sigmoid_kernel(float* input, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_kernel(float* input, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(input[idx]);
    }
}

__global__ void pow_kernel(float* a, float exponent, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = powf(a[idx], exponent);
    }
}

__global__ void sum_all_kernel(float* input, float* out, int64_t size) {
    __shared__ float cache[256];
    int64_t tid = threadIdx.x;
    int64_t i = blockIdx.x * blockDim.x + tid;
    float temp = 0.0f;
    if (i < size) {
        temp = input[i];
    }
    cache[tid] = temp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, cache[0]);
    }
}

__global__ void transpose_kernel(float* input, float* out, int64_t rows, int64_t cols) {
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        out[x * rows + y] = input[y * cols + x];
    }
}

namespace libshit::core {

// ========== Existing Ops ==========

SHIT_API ShitTensor* relu(ShitTensor* tensor) {
    int threads_per_block = 256;
    int64_t blocks = (tensor->size() + threads_per_block - 1) / threads_per_block;
    ShitTensor* out_tensor = new ShitTensor(tensor->tensor_shape());

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<ReLUNode>(tensor, out_tensor);
        ShitTape::instance().push_node(std::move(node));
    }

    relu_kernel<<<blocks, threads_per_block>>>(tensor->gpu_ptr(), out_tensor->gpu_ptr(), tensor->size());
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

    int a_rows = static_cast<int>(a_shape[0]);
    int a_cols = static_cast<int>(a_shape[1]);
    int b_cols = static_cast<int>(b_shape[1]);

    ShitTensor* out_tensor = new ShitTensor({a_rows, b_cols});

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<MatMulNode>(tensora, tensorb, out_tensor);
        ShitTape::instance().push_node(std::move(node));
    }

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (b_cols + threads_per_block.x - 1) / threads_per_block.x,
        (a_rows + threads_per_block.y - 1) / threads_per_block.y
    );

    mm_kernel<<<blocks_per_grid, threads_per_block>>>(
        tensora->gpu_ptr(), tensorb->gpu_ptr(), out_tensor->gpu_ptr(), a_rows, a_cols, b_cols);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out_tensor;
}

SHIT_API ShitTensor* add(ShitTensor* tensora, ShitTensor* tensorb) {
    const std::vector<int64_t>& a_shape = tensora->get_shape();
    const std::vector<int64_t>& b_shape = tensorb->get_shape();

    if (a_shape != b_shape) {
        std::cerr << "LIBSHIT ADD ERROR: Shape mismatch! "
                  << "Cannot add (" << a_shape[0] << "x" << a_shape[1] << ") and ("
                  << b_shape[0] << "x" << b_shape[1] << ")" << std::endl;
        exit(1);
    }

    int a_rows = static_cast<int>(a_shape[0]);
    int a_cols = static_cast<int>(a_shape[1]);
    int b_cols = static_cast<int>(b_shape[1]);

    ShitTensor* out_tensor = new ShitTensor({a_rows, b_cols});

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<AddNode>(tensora, tensorb, out_tensor);
        ShitTape::instance().push_node(std::move(node));
    }

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (b_cols + threads_per_block.x - 1) / threads_per_block.x,
        (a_rows + threads_per_block.y - 1) / threads_per_block.y
    );

    add_kernel<<<blocks_per_grid, threads_per_block>>>(
        tensora->gpu_ptr(), tensorb->gpu_ptr(), out_tensor->gpu_ptr(), a_rows, a_cols, b_cols);
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

// ========== New Ops ==========

SHIT_API ShitTensor* subtract(ShitTensor* a, ShitTensor* b) {
    if (a->get_shape() != b->get_shape()) {
        std::cerr << "LIBSHIT SUBTRACT ERROR: Shape mismatch!" << std::endl;
        exit(1);
    }
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<SubtractNode>(a, b, out);
        ShitTape::instance().push_node(std::move(node));
    }

    subtract_kernel<<<blocks, threads>>>(a->gpu_ptr(), b->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* multiply(ShitTensor* a, ShitTensor* b) {
    if (a->get_shape() != b->get_shape()) {
        std::cerr << "LIBSHIT MULTIPLY ERROR: Shape mismatch!" << std::endl;
        exit(1);
    }
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<MultiplyNode>(a, b, out);
        ShitTape::instance().push_node(std::move(node));
    }

    multiply_kernel<<<blocks, threads>>>(a->gpu_ptr(), b->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* negate(ShitTensor* a) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<NegateNode>(a, out);
        ShitTape::instance().push_node(std::move(node));
    }

    negate_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* sigmoid(ShitTensor* a) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<SigmoidNode>(a, out);
        ShitTape::instance().push_node(std::move(node));
    }

    sigmoid_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* tanh_act(ShitTensor* a) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<TanhNode>(a, out);
        ShitTape::instance().push_node(std::move(node));
    }

    tanh_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* pow_op(ShitTensor* a, float exponent) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<PowNode>(a, exponent, out);
        ShitTape::instance().push_node(std::move(node));
    }

    pow_kernel<<<blocks, threads>>>(a->gpu_ptr(), exponent, out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* sum_all(ShitTensor* a) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor({1});

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<SumAllNode>(a, out);
        ShitTape::instance().push_node(std::move(node));
    }

    sum_all_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* transpose(ShitTensor* a) {
    const auto& shape = a->get_shape();
    if (shape.size() != 2) {
        std::cerr << "LIBSHIT TRANSPOSE ERROR: Only 2D tensors supported." << std::endl;
        exit(1);
    }
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    ShitTensor* out = new ShitTensor({cols, rows});

    if (ShitTape::instance().is_active()) {
        auto node = std::make_unique<TransposeNode>(a, out);
        ShitTape::instance().push_node(std::move(node));
    }

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    transpose_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), rows, cols);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* reshape(ShitTensor* a, const std::vector<int64_t>& new_shape) {
    int64_t new_total = 1;
    for (auto d : new_shape) new_total *= d;
    if (new_total != a->size()) {
        std::cerr << "LIBSHIT RESHAPE ERROR: Element count mismatch!" << std::endl;
        exit(1);
    }
    // Reshape is just a view — no kernel needed, no tape node needed
    ShitTensor* out = new ShitTensor(new_shape);
    out->set_data(a->cpu_ptr());
    // Copy GPU data too
    cudaMemcpy(out->gpu_ptr(), a->gpu_ptr(), a->size() * sizeof(float), cudaMemcpyDeviceToDevice);
    return out;
}

} // namespace libshit::core