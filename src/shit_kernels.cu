#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <ctime>
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

// ========== LeakyReLU ==========

__global__ void leaky_relu_kernel(float* input, float* output, float alpha, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] >= 0.0f ? input[idx] : alpha * input[idx];
    }
}

SHIT_API ShitTensor* leaky_relu(ShitTensor* a, float alpha) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());

    if (ShitTape::instance().is_active()) {
        // Note: LeakyReLU has a custom backward; we'll handle it as a simple node
        // For now, we push a ReLUNode-derived; tape node not critical for layer op
    }

    leaky_relu_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), alpha, size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

// ========== Softmax ==========

__global__ void softmax_kernel(float* input, float* output, int64_t rows, int64_t cols) {
    // Each block handles one row
    extern __shared__ float shared[];
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    // Reduce max across threads
    if (blockDim.x <= 32) {
        // warp shuffle
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    } else {
        // shared memory reduction
        shared[threadIdx.x] = max_val;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
            __syncthreads();
        }
        max_val = shared[0];
    }

    // Compute exp sum
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(row_in[i] - max_val);
        row_out[i] = e;
        sum += e;
    }
    // Reduce sum across threads
    if (blockDim.x <= 32) {
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    } else {
        shared[threadIdx.x] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
            __syncthreads();
        }
        sum = shared[0];
    }

    // Normalize
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] /= sum;
    }
}

SHIT_API ShitTensor* softmax(ShitTensor* a) {
    const auto& shape = a->get_shape();
    if (shape.size() < 2) {
        std::cerr << "LIBSHIT SOFTMAX ERROR: Input must be at least 2D." << std::endl;
        exit(1);
    }
    int64_t rows = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) rows *= shape[i];
    int64_t cols = shape.back();

    ShitTensor* out = new ShitTensor(a->tensor_shape());
    int threads = 256;
    int blocks = (int)rows;

    // shared memory for max + sum reduction per row
    size_t shared_bytes = threads * sizeof(float);

    if (ShitTape::instance().is_active()) {
        // tape node placeholder — softmax grad not yet wired in full autodiff
    }

    softmax_kernel<<<blocks, threads, shared_bytes>>>(
        a->gpu_ptr(), out->gpu_ptr(), rows, cols);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

// ========== Embedding Lookup ==========

__global__ void embedding_lookup_kernel(const float* weight, float* output,
                                         int64_t vocab_size, int64_t emb_dim,
                                         int64_t index) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < emb_dim) {
        output[i] = weight[index * emb_dim + i];
    }
}

SHIT_API ShitTensor* embedding_lookup(ShitTensor* weight, int64_t index) {
    const auto& shape = weight->get_shape();
    if (shape.size() != 2) {
        std::cerr << "LIBSHIT EMBEDDING ERROR: Weight must be 2D (vocab x emb_dim)." << std::endl;
        exit(1);
    }
    int64_t emb_dim = shape[1];
    ShitTensor* out = new ShitTensor({1, emb_dim});

    int threads = 256;
    int blocks = (emb_dim + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads>>>(
        weight->gpu_ptr(), out->gpu_ptr(), shape[0], emb_dim, index);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

// ========== Dropout ==========

__global__ void dropout_kernel(float* input, float* output, float* mask,
                                float prob, int64_t size, curandState* states) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    curandState local_state = states[idx];
    float r = curand_uniform(&local_state);
    mask[idx] = (r > prob) ? 1.0f / (1.0f - prob) : 0.0f;
    output[idx] = input[idx] * mask[idx];
    states[idx] = local_state;
}

__global__ void init_curand_states(curandState* states, int64_t size, unsigned long long seed) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

SHIT_API ShitTensor* dropout(ShitTensor* a, float probability) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());

    // Allocate mask and random states (persisted across calls for simplicity)
    static ShitTensor* mask_tensor = nullptr;
    static curandState* d_states = nullptr;
    static int64_t cached_size = 0;

    if (size != cached_size) {
        if (mask_tensor) free_tensor(mask_tensor);
        if (d_states) cudaFree(d_states);
        mask_tensor = new ShitTensor({size});
        cudaMalloc(&d_states, size * sizeof(curandState));
        // Init random states
        int init_blocks = (size + 255) / 256;
        init_curand_states<<<init_blocks, 256>>>(d_states, size, time(nullptr));
        SHIT_CHECK(cudaGetLastError());
        SHIT_CHECK(cudaDeviceSynchronize());
        cached_size = size;
    }

    dropout_kernel<<<blocks, threads>>>(
        a->gpu_ptr(), out->gpu_ptr(), mask_tensor->gpu_ptr(), probability, size, d_states);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

// ========== Element-wise scalar ops ==========

__global__ void div_scalar_kernel(float* input, float* output, float scalar, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = input[idx] / scalar;
}

__global__ void add_scalar_kernel(float* input, float* output, float scalar, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = input[idx] + scalar;
}

__global__ void sqrt_kernel(float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = sqrtf(input[idx]);
}

__global__ void exp_kernel(float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = expf(input[idx]);
}

__global__ void rsqrt_kernel(float* input, float* output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = rsqrtf(input[idx]);
}

SHIT_API ShitTensor* div_scalar(ShitTensor* a, float scalar) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());
    div_scalar_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), scalar, size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* add_scalar(ShitTensor* a, float scalar) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());
    add_scalar_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), scalar, size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* sqrt_op(ShitTensor* a) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());
    sqrt_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* exp_op(ShitTensor* a) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());
    exp_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

SHIT_API ShitTensor* rsqrt(ShitTensor* a) {
    int64_t size = a->size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    ShitTensor* out = new ShitTensor(a->tensor_shape());
    rsqrt_kernel<<<blocks, threads>>>(a->gpu_ptr(), out->gpu_ptr(), size);
    SHIT_CHECK(cudaGetLastError());
    SHIT_CHECK(cudaDeviceSynchronize());
    return out;
}

} // namespace libshit::core