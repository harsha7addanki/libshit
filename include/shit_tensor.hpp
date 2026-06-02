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
        int64_t total_elements; // also js there

        float* h_data; // CPU data
        float* d_data; // GPU data for HIP/CUDA

        void compute_strides();
        void initialize(InitType init);

    public:
        ShitTensor(std::initializer_list<int64_t> dims, InitType init = InitType::None);
        ShitTensor(const std::vector<int64_t>& dims, InitType init = InitType::None);
        ~ShitTensor();

        // dont copy
        ShitTensor(const ShitTensor&) = delete;
        ShitTensor& operator=(const ShitTensor&) = delete;

        // non unified memory
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
            h_data = h_ptr; // set the host pointer to the provided data
            // TODO: add check if the data is on gpu so we dont put data on gpu if dev didnt want it there
            to_gpu(); // transfer this new data to the GPU
        }
    };

    SHIT_API ShitTensor* relu(ShitTensor* tensor);
    SHIT_API ShitTensor* matmul(ShitTensor* tensora, ShitTensor* tensorb);
    SHIT_API ShitTensor* add(ShitTensor* tensora, ShitTensor* tensorb);
    SHIT_API ShitTensor* mse(ShitTensor* pred, ShitTensor* target);

    SHIT_API void free_tensor(ShitTensor* tensor);
}
#endif