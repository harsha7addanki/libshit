#include "../include/shit_tensor.hpp"
#include "../include/shit_gradients.hpp"
#include <iostream>
#include <cmath>

static bool close_enough(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

static void print_matrix(const char* name, libshit::core::ShitTensor* t) {
    t->to_cpu();
    std::cout << name << " = [";
    for (int64_t i = 0; i < t->size(); ++i) {
        std::cout << (*t)[i];
        if (i + 1 < t->size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    using namespace libshit::core;

    std::cout << "Starting gradient loss scaling test..." << std::endl;

    ShitTensor* X = new ShitTensor({1, 2});
    ShitTensor* W1 = new ShitTensor({2, 2});
    ShitTensor* W2 = new ShitTensor({2, 1});

    (*X)[0] = 1.0f;
    (*X)[1] = 1.0f;

    (*W1)[0] = 1.0f;   (*W1)[1] = -2.0f;
    (*W1)[2] = 2.0f;   (*W1)[3] = 1.0f;

    (*W2)[0] = 2.0f;
    (*W2)[1] = -0.5f;

    X->to_gpu();
    W1->to_gpu();
    W2->to_gpu();

    ShitTape::instance().start();
    ShitTensor* hidden = matmul(X, W1);
    ShitTensor* hidden_relu = relu(hidden);
    ShitTensor* output = matmul(hidden_relu, W2);
    ShitTensor* target = new ShitTensor({1});
    (*target)[0] = 7.0f;
    target->to_gpu();
    ShitTensor* loss = mse(output, target);
    ShitTape::instance().stop();

    if (ShitTape::instance().get_nodes().size() != 4) {
        std::cerr << "Gradient tape did not record all network nodes." << std::endl;
        return 1;
    }

    std::cout << "Network forward pass complete." << std::endl;
    print_matrix("W1 before", W1);
    print_matrix("W2 before", W2);

    ShitGradientRegistry registry;
    ShitTensor* grad_loss = registry.get_grad(loss);
    const float loss_scale = 5.0f;
    (*grad_loss)[0] = loss_scale;
    grad_loss->to_gpu();

    ShitTape::instance().backward(registry);

    ShitTensor* grad_W1 = registry.get_grad(W1);
    ShitTensor* grad_W2 = registry.get_grad(W2);

    grad_W1->to_cpu();
    grad_W2->to_cpu();

    const float expected_grad_W1[4] = {-20.0f, 0.0f, -20.0f, 0.0f};
    const float expected_grad_W2[2] = {-30.0f, 0.0f};

    bool passed = true;
    for (int64_t i = 0; i < grad_W1->size(); ++i) {
        if (!close_enough((*grad_W1)[i], expected_grad_W1[i])) {
            std::cerr << "grad_W1[" << i << "] = " << (*grad_W1)[i]
                      << " expected " << expected_grad_W1[i] << std::endl;
            passed = false;
        }
    }
    for (int64_t i = 0; i < grad_W2->size(); ++i) {
        if (!close_enough((*grad_W2)[i], expected_grad_W2[i])) {
            std::cerr << "grad_W2[" << i << "] = " << (*grad_W2)[i]
                      << " expected " << expected_grad_W2[i] << std::endl;
            passed = false;
        }
    }

    std::cout << "Applying scaled weight updates (lr=0.1)..." << std::endl;
    grad_W1->to_gpu();
    grad_W2->to_gpu();
    update_weights(W1, grad_W1, 0.1f);
    update_weights(W2, grad_W2, 0.1f);

    std::cout << "W1 after update:" << std::endl;
    print_matrix("W1 after", W1);
    std::cout << "W2 after update:" << std::endl;
    print_matrix("W2 after", W2);

    if (passed) {
        std::cout << "Gradient loss scaling test passed." << std::endl;
    } else {
        std::cerr << "Gradient loss scaling test failed." << std::endl;
    }

    free_tensor(X);
    free_tensor(W1);
    free_tensor(W2);
    free_tensor(hidden);
    free_tensor(hidden_relu);
    free_tensor(output);
    free_tensor(target);
    free_tensor(loss);
    free_tensor(grad_loss);
    free_tensor(grad_W1);
    free_tensor(grad_W2);

    return passed ? 0 : 1;
}
