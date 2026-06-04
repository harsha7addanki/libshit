#ifndef SHIT_TENSOR_H
#define SHIT_TENSOR_H

#ifdef _WIN32
    #ifdef LIBSHIT_EXPORTS
        #define SHIT_API __declspec(dllexport)
    #else
        #define SHIT_API __declspec(dllimport)
    #endif
#else
    #define SHIT_API
#endif

#include <cuda_runtime.h>
#include <vector>
#include <initializer_list>
#include <cstdint>

namespace libshit::core{
    // a shitty cuda tensor
    class SHIT_API ShitTensor {
    public:
        enum class InitType { None, XavierUniform, RandomUniform };
    private:
        std::vector<int64_t> shape; // its exactly like everything else
        std::vector<int64_t> strides; // precomputed stride for indexing
        int64_t total_elements = 0; // also js there

        float* h_data = nullptr; // cpu pointer
        float* d_data = nullptr; // gpu pointer for cuda(or hip if i ever compile it with that)
        bool owns_h_data = true;

        void compute_strides();
        void initialize(InitType init);

    public:
        ShitTensor(std::initializer_list<int64_t> dims, InitType init = InitType::None);
        ShitTensor(const std::vector<int64_t>& dims, InitType init = InitType::None);
        ~ShitTensor();

        // dont copy
        ShitTensor(const ShitTensor&) = delete;
        ShitTensor& operator=(const ShitTensor&) = delete;

        // non unified memory(better)
        void to_gpu();
        void to_cpu();

        // Core Properties
        int64_t size() const { return total_elements; }
        const std::vector<int64_t>& get_shape() const { return shape; }
        float* cpu_ptr() { return h_data; }
        float* gpu_ptr() { return d_data; }
        std::vector<int64_t> tensor_shape() const { return shape; }

        float& operator[](int64_t idx) { 
            return h_data[idx]; 
        }

        const float& operator[](int64_t idx) const { 
            return h_data[idx]; 
        }


        
        float& operator()(std::initializer_list<int64_t> indices);
        void set_data(float* h_ptr) {
            if (owns_h_data && h_data != nullptr) {
                cudaFreeHost(h_data);
                h_data = nullptr;
            }
            h_data = h_ptr; // set the host pointer to the provided data
            owns_h_data = false;
            // TODO: add check if the data is on gpu so we dont put data on gpu if they didnt want it there
            to_gpu(); // transfer this new data to the GPU
        }
    };

    // Existing ops
    SHIT_API ShitTensor* relu(ShitTensor* tensor);
    SHIT_API ShitTensor* matmul(ShitTensor* tensora, ShitTensor* tensorb);
    SHIT_API ShitTensor* add(ShitTensor* tensora, ShitTensor* tensorb);
    SHIT_API ShitTensor* mse(ShitTensor* pred, ShitTensor* target);

    // New element-wise ops
    SHIT_API ShitTensor* subtract(ShitTensor* a, ShitTensor* b);
    SHIT_API ShitTensor* multiply(ShitTensor* a, ShitTensor* b);
    SHIT_API ShitTensor* negate(ShitTensor* a);
    SHIT_API ShitTensor* sigmoid(ShitTensor* a);
    SHIT_API ShitTensor* tanh_act(ShitTensor* a);
    SHIT_API ShitTensor* pow_op(ShitTensor* a, float exponent);

    // New reduction ops
    SHIT_API ShitTensor* sum_all(ShitTensor* a);

    // New matrix ops
    SHIT_API ShitTensor* transpose(ShitTensor* a);
    SHIT_API ShitTensor* reshape(ShitTensor* a, const std::vector<int64_t>& new_shape);

        // Additional activations
        SHIT_API ShitTensor* leaky_relu(ShitTensor* a, float alpha);
        SHIT_API ShitTensor* softmax(ShitTensor* a);

        // Embedding / lookup
        SHIT_API ShitTensor* embedding_lookup(ShitTensor* weight, int64_t index);

        // Dropout
        SHIT_API ShitTensor* dropout(ShitTensor* a, float probability);

        // Element-wise helpers for normalisation layers
        SHIT_API ShitTensor* div_scalar(ShitTensor* a, float scalar);
        SHIT_API ShitTensor* add_scalar(ShitTensor* a, float scalar);
        SHIT_API ShitTensor* sqrt_op(ShitTensor* a);
        SHIT_API ShitTensor* exp_op(ShitTensor* a);
        SHIT_API ShitTensor* rsqrt(ShitTensor* a);

        SHIT_API void free_tensor(ShitTensor* tensor);
}
#endif