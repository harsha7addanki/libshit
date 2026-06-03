#include "../include/shit_tensor.hpp"
#include "../include/shit_gradients.hpp"
#include <iostream>
#include <cmath>

int main() {
    using namespace libshit::core;

    auto print_matrix = [&](const char* name, ShitTensor* t) {
        t->to_cpu();
        std::cout << name << " = [";
        for (int64_t i = 0; i < t->size(); ++i) {
            std::cout << (*t)[i];
            if (i + 1 < t->size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    };

    std::cout << "Starting gradient tape test..." << std::endl;

    ShitTensor* A = new ShitTensor({2, 2});
    ShitTensor* B = new ShitTensor({2, 2});

    (*A)[0] = 1.0f;  (*A)[1] = -2.0f;
    (*A)[2] = 3.0f;  (*A)[3] = -4.0f;

    (*B)[0] = 5.0f;  (*B)[1] = 6.0f;
    (*B)[2] = 7.0f;  (*B)[3] = -8.0f;

    A->to_gpu();
    B->to_gpu();

    ShitTape::instance().start();
    ShitTensor* mat_out = matmul(A, B);
    ShitTensor* relu_out = relu(mat_out);
    ShitTape::instance().stop();

    if (ShitTape::instance().get_nodes().size() != 2) {
        std::cerr << "Gradient tape did not record the expected number of nodes." << std::endl;
        return 1;
    }

    ShitGradientRegistry registry;
    ShitTensor* grad_out = registry.get_grad(relu_out);

    for (int64_t i = 0; i < grad_out->size(); ++i) {
        (*grad_out)[i] = 1.0f;
    }
    grad_out->to_gpu();

    ShitTape::instance().backward(registry);

    ShitTensor* grad_A = registry.get_grad(A);
    ShitTensor* grad_B = registry.get_grad(B);

    grad_A->to_cpu();
    grad_B->to_cpu();

    std::cout << "Weights before update:" << std::endl;
    print_matrix("A", A);
    print_matrix("B", B);

    const float expected_grad_A[4] = {6.0f, -8.0f, 6.0f, -8.0f};
    const float expected_grad_B[4] = {0.0f, 4.0f, 0.0f, -6.0f};

    bool passed = true;

    for (int64_t i = 0; i < grad_A->size(); ++i) {
        float actual = (*grad_A)[i];
        if (fabs(actual - expected_grad_A[i]) > 1e-4f) {
            std::cerr << "grad_A[" << i << "] = " << actual
                      << " expected " << expected_grad_A[i] << std::endl;
            passed = false;
        }
    }

    for (int64_t i = 0; i < grad_B->size(); ++i) {
        float actual = (*grad_B)[i];
        if (fabs(actual - expected_grad_B[i]) > 1e-4f) {
            std::cerr << "grad_B[" << i << "] = " << actual
                      << " expected " << expected_grad_B[i] << std::endl;
            passed = false;
        }
    }

    std::cout << "Applying weight updates with learning rate 0.1..." << std::endl;
    grad_A->to_gpu();
    grad_B->to_gpu();
    update_weights(A, grad_A, 0.1f);
    update_weights(B, grad_B, 0.1f);

    std::cout << "Weights after update:" << std::endl;
    print_matrix("A", A);
    print_matrix("B", B);

    if (passed) {
        std::cout << "Gradient tape test passed." << std::endl;
    } else {
        std::cerr << "Gradient tape test failed." << std::endl;
    }

    libshit::core::free_tensor(A);
    libshit::core::free_tensor(B);
    libshit::core::free_tensor(mat_out);
    libshit::core::free_tensor(relu_out);
    // grad_out, grad_A, and grad_B are owned by registry and will be freed by its destructor.

    return passed ? 0 : 1;
}
