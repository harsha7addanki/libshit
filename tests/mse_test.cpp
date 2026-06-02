#include "../include/shit_tensor.hpp"
#include <iostream>
#include <cmath>

static bool close_enough(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    using namespace libshit::core;

    ShitTensor* prediction = new ShitTensor({3});
    ShitTensor* target = new ShitTensor({3});

    (*prediction)[0] = 1.0f;
    (*prediction)[1] = 2.0f;
    (*prediction)[2] = 3.0f;

    (*target)[0] = 1.5f;
    (*target)[1] = 1.0f;
    (*target)[2] = 2.5f;

    prediction->to_gpu();
    target->to_gpu();

    ShitTensor* loss = mse(prediction, target);
    loss->to_cpu();

    float actual = loss->cpu_ptr()[0];
    float expected = (1.0f - 1.5f) * (1.0f - 1.5f)
                   + (2.0f - 1.0f) * (2.0f - 1.0f)
                   + (3.0f - 2.5f) * (3.0f - 2.5f);

    std::cout << "MSE loss = " << actual << std::endl;

    bool passed = close_enough(actual, expected);
    if (!passed) {
        std::cerr << "MSE mismatch: expected " << expected << " but got " << actual << std::endl;
    }

    libshit::core::free_tensor(prediction);
    libshit::core::free_tensor(target);
    libshit::core::free_tensor(loss);

    return passed ? 0 : 1;
}
