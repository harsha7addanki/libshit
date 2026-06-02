#include "../include/shit_tensor.hpp"
#include <iostream>
#include <cmath>

static bool close_enough(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

static void print_values(const char* prefix, libshit::core::ShitTensor* tensor) {
    tensor->to_cpu();
    std::cout << prefix << " [";
    for (int64_t i = 0; i < tensor->size(); ++i) {
        std::cout << (*tensor)[i];
        if (i + 1 < tensor->size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    bool all_passed = true;

    // ReLU test
    std::cout << "Running ReLU unit test..." << std::endl;
    libshit::core::ShitTensor* input = new libshit::core::ShitTensor({4});
    (*input)[0] = -10.5f;
    (*input)[1] = 42.0f;
    (*input)[2] = -0.01f;
    (*input)[3] = 7.7f;

    input->to_gpu();
    libshit::core::ShitTensor* relu_out = libshit::core::relu(input);
    relu_out->to_cpu();

    const float expected_relu[4] = {0.0f, 42.0f, 0.0f, 7.7f};
    for (int64_t i = 0; i < relu_out->size(); ++i) {
        if (!close_enough((*relu_out)[i], expected_relu[i])) {
            std::cerr << "ReLU mismatch at index " << i << ": got " << (*relu_out)[i]
                      << " expected " << expected_relu[i] << std::endl;
            all_passed = false;
        }
    }

    print_values("ReLU output", relu_out);
    libshit::core::free_tensor(input);

    // MatMul test
    std::cout << "\nRunning MatMul unit test..." << std::endl;
    libshit::core::ShitTensor* A = new libshit::core::ShitTensor({2, 2});
    libshit::core::ShitTensor* B = new libshit::core::ShitTensor({2, 2});

    (*A)[0] = 1.0f; (*A)[1] = 2.0f;
    (*A)[2] = 3.0f; (*A)[3] = 4.0f;
    (*B)[0] = 5.0f; (*B)[1] = 6.0f;
    (*B)[2] = 7.0f; (*B)[3] = 8.0f;

    A->to_gpu();
    B->to_gpu();
    libshit::core::ShitTensor* matmul_out = libshit::core::matmul(A, B);
    matmul_out->to_cpu();

    const float expected_matmul[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    for (int64_t i = 0; i < matmul_out->size(); ++i) {
        if (!close_enough((*matmul_out)[i], expected_matmul[i])) {
            std::cerr << "MatMul mismatch at index " << i << ": got " << (*matmul_out)[i]
                      << " expected " << expected_matmul[i] << std::endl;
            all_passed = false;
        }
    }

    print_values("MatMul output", matmul_out);

    if (all_passed) {
        std::cout << "\nAll tests passed." << std::endl;
    } else {
        std::cerr << "\nOne or more tests failed." << std::endl;
    }

    libshit::core::free_tensor(relu_out);
    libshit::core::free_tensor(A);
    libshit::core::free_tensor(B);
    libshit::core::free_tensor(matmul_out);

    return all_passed ? 0 : 1;
}
