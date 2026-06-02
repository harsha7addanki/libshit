#include <cuda_runtime.h>
#include <numeric>
#include <iostream>
#include <random>
#include <cstring>
#include "../include/shit_tensor.hpp"
#include "../include/shit_errors.hpp"

// computes strides for each dimension
namespace libshit::core{

    void ShitTensor::compute_strides() {
        strides.resize(shape.size());
        int64_t current_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
    }
    
    // Initializer list constructor
    ShitTensor::ShitTensor(std::initializer_list<int64_t> dims, InitType init) : shape(dims) {
        // find total elements
        total_elements = 1;
        for (auto d : shape) {
            total_elements *= d;
        }
    
        compute_strides();
    
        size_t bytes = total_elements * sizeof(float);
        SHIT_CHECK(cudaHostAlloc((void**)&h_data, bytes, cudaHostAllocDefault));
        std::memset(h_data, 0, bytes);
        SHIT_CHECK(cudaMalloc((void**)&d_data, bytes));
        initialize(init);
        to_gpu();
    }
    
    ShitTensor::ShitTensor(const std::vector<int64_t>& dims, InitType init) {
        shape = dims;
        total_elements = 1;
        for (auto d : shape) {
            total_elements *= d;
        }
        
        compute_strides();
        size_t bytes = total_elements * sizeof(float);
        SHIT_CHECK(cudaHostAlloc((void**)&h_data, bytes, cudaHostAllocDefault));
        std::memset(h_data, 0, bytes);
        SHIT_CHECK(cudaMalloc((void**)&d_data, bytes));
        initialize(init);
        to_gpu();
    }
    
    ShitTensor::~ShitTensor() {
        if (owns_h_data && h_data != nullptr) {
            SHIT_CHECK(cudaFreeHost(h_data));
        }
        if (d_data != nullptr) {
            SHIT_CHECK(cudaFree(d_data));
        }
    }
    
    // flats the cordinates by using the stride
    float& ShitTensor::operator()(std::initializer_list<int64_t> indices) {
        auto it = indices.begin();
        int64_t flat_index = 0;
        
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t idx = *it++;
            // multipy by stride
            flat_index += idx * strides[i];
        }
        return h_data[flat_index];
    }
    
    // pretty obvious what theese do
    void ShitTensor::initialize(InitType init) {
        if (init == InitType::None) {
            return;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        float limit = 0.1f;

        if (init == InitType::XavierUniform && shape.size() == 2) {
            float fan_in = static_cast<float>(shape[0]);
            float fan_out = static_cast<float>(shape[1]);
            limit = std::sqrt(6.0f / (fan_in + fan_out));
        }

        std::uniform_real_distribution<float> dist(-limit, limit);
        for (int64_t i = 0; i < total_elements; ++i) {
            h_data[i] = dist(gen);
        }
    }

    void ShitTensor::to_gpu() {
        size_t bytes = total_elements * sizeof(float);
        SHIT_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    }
    
    void ShitTensor::to_cpu() {
        size_t bytes = total_elements * sizeof(float);
        SHIT_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    }
    
    SHIT_API void free_tensor(ShitTensor* tensor) {
        if(tensor != nullptr){
            delete tensor;
        }
    }
}