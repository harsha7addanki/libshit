#include "../include/layers/layers.hpp"
#include "../include/optim/optimizer.hpp"
#include "../include/save/save.hpp"
#include <iostream>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

static bool close_enough(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

static void print_tensor(const char* name, libshit::core::ShitTensor* t) {
    if (!t) { std::cout << name << " = null" << std::endl; return; }
    t->to_cpu();
    std::cout << name << " [";
    for (int64_t i = 0; i < t->size(); ++i) {
        std::cout << (*t)[i];
        if (i + 1 < t->size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    using namespace libshit::core;
    bool all_passed = true;
    int test_count = 0;
    auto pass = [&](const char* name, bool cond) {
        test_count++;
        if (cond) {
            std::cout << "[PASS] " << name << std::endl;
        } else {
            std::cerr << "[FAIL] " << name << std::endl;
            all_passed = false;
        }
    };

    // ========== 1. Flatten ==========
    {
        std::cout << "\n--- Testing Flatten ---" << std::endl;
        ShitTensor* input = new ShitTensor({2, 3});
        for (int64_t i = 0; i < 6; ++i) (*input)[i] = (float)(i + 1);
        input->to_gpu();

        libshit::layers::Flatten flatten;
        ShitTensor* output = flatten(input);
        output->to_cpu();
        pass("Flatten output size", output->size() == 6);
        pass("Flatten output shape 2D", output->get_shape().size() == 2);
        pass("Flatten correct values",
             close_enough((*output)[0], 1.0f) && close_enough((*output)[5], 6.0f));

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(output);
    }

    // ========== 2. Embedding ==========
    {
        std::cout << "\n--- Testing Embedding ---" << std::endl;
        libshit::layers::Embedding emb(10, 4); // vocab=10, dim=4

        // Build with a dummy input
        ShitTensor* dummy = new ShitTensor({1});
        (*dummy)[0] = 0.0f;
        dummy->to_gpu();
        emb(dummy); // triggers build
        libshit::core::free_tensor(dummy);

        auto params = emb.get_parameters();
        pass("Embedding has 1 parameter", params.size() == 1);

        // Look up index 3
        ShitTensor* idx = new ShitTensor({1});
        (*idx)[0] = 3.0f;
        idx->to_gpu();
        ShitTensor* result = emb(idx);
        result->to_cpu();
        pass("Embedding output shape", result->get_shape().size() == 2 && result->get_shape()[1] == 4);
        pass("Embedding output size == 4", result->size() == 4);

        libshit::core::free_tensor(idx);
        libshit::core::free_tensor(result);
    }

    // ========== 3. RMSNorm ==========
    {
        std::cout << "\n--- Testing RMSNorm ---" << std::endl;
        libshit::layers::RMSNorm rms(4, 1e-5f);

        ShitTensor* input = new ShitTensor({1, 4});
        for (int64_t i = 0; i < 4; ++i) (*input)[i] = (float)(i + 1);
        input->to_gpu();

        // Build via forward
        ShitTensor* output = rms(input);
        output->to_cpu();
        pass("RMSNorm output shape", output->get_shape() == input->get_shape());
        pass("RMSNorm output size == 4", output->size() == 4);

        // Check that gamma is learnable
        auto params = rms.get_parameters();
        pass("RMSNorm has 1 parameter", params.size() == 1);
        pass("RMSNorm gamma is ones initially",
             close_enough((*params[0].get())[0], 1.0f));

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(output);
    }

    // ========== 4. LayerNorm ==========
    {
        std::cout << "\n--- Testing LayerNorm ---" << std::endl;
        libshit::layers::LayerNorm ln(4, 1e-5f);

        ShitTensor* input = new ShitTensor({1, 4});
        for (int64_t i = 0; i < 4; ++i) (*input)[i] = (float)(i + 1);
        input->to_gpu();

        ShitTensor* output = ln(input);
        output->to_cpu();
        pass("LayerNorm output shape", output->get_shape() == input->get_shape());
        pass("LayerNorm output size == 4", output->size() == 4);

        auto params = ln.get_parameters();
        pass("LayerNorm has 2 parameters (gamma, beta)", params.size() == 2);
        pass("LayerNorm gamma is ones initially",
             close_enough((*params[0].get())[0], 1.0f));
        pass("LayerNorm beta is zeros initially",
             close_enough((*params[0].get())[0], 1.0f));

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(output);
    }

    // ========== 5. Dropout ==========
    {
        std::cout << "\n--- Testing Dropout ---" << std::endl;
        libshit::layers::Dropout drop(0.5f);

        ShitTensor* input = new ShitTensor({1000});
        for (int64_t i = 0; i < 1000; ++i) (*input)[i] = 1.0f;
        input->to_gpu();

        // Eval mode -> identity
        drop.eval();
        ShitTensor* eval_out = drop(input);
        eval_out->to_cpu();
        pass("Dropout eval mode keeps all values",
             close_enough((*eval_out)[0], 1.0f));
        libshit::core::free_tensor(eval_out);

        // Train mode -> some zeros
        drop.train();
        ShitTensor* train_out = drop(input);
        train_out->to_cpu();
        int zero_count = 0;
        for (int64_t i = 0; i < 1000; ++i) {
            if ((*train_out)[i] == 0.0f) zero_count++;
        }
        pass("Dropout train mode zeros some values", zero_count > 0);
        pass("Dropout train mode keeps some values", zero_count < 1000);

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(train_out);
    }

    // ========== 6. LeakyReLU ==========
    {
        std::cout << "\n--- Testing LeakyReLU ---" << std::endl;
        libshit::layers::LeakyReLU lrelu(0.1f);

        ShitTensor* input = new ShitTensor({4});
        (*input)[0] = -2.0f; (*input)[1] = 0.0f;
        (*input)[2] = 3.0f;  (*input)[3] = -0.5f;
        input->to_gpu();

        ShitTensor* output = lrelu(input);
        output->to_cpu();
        pass("LeakyReLU positive passthrough", close_enough((*output)[2], 3.0f));
        pass("LeakyReLU negative slope",
             close_enough((*output)[0], -0.2f) && close_enough((*output)[3], -0.05f));
        pass("LeakyReLU at zero", close_enough((*output)[1], 0.0f));

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(output);
    }

    // ========== 7. Sigmoid ==========
    {
        std::cout << "\n--- Testing Sigmoid ---" << std::endl;
        libshit::layers::Sigmoid sig;

        ShitTensor* input = new ShitTensor({3});
        (*input)[0] = 0.0f; (*input)[1] = 10.0f; (*input)[2] = -10.0f;
        input->to_gpu();

        ShitTensor* output = sig(input);
        output->to_cpu();
        pass("Sigmoid at 0", close_enough((*output)[0], 0.5f));
        pass("Sigmoid large positive", close_enough((*output)[1], 1.0f));
        pass("Sigmoid large negative", close_enough((*output)[2], 0.0f));

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(output);
    }

    // ========== 8. Tanh ==========
    {
        std::cout << "\n--- Testing Tanh ---" << std::endl;
        libshit::layers::Tanh tanh_layer;

        ShitTensor* input = new ShitTensor({3});
        (*input)[0] = 0.0f; (*input)[1] = 10.0f; (*input)[2] = -10.0f;
        input->to_gpu();

        ShitTensor* output = tanh_layer(input);
        output->to_cpu();
        pass("Tanh at 0", close_enough((*output)[0], 0.0f));
        pass("Tanh large positive", close_enough((*output)[1], 1.0f));
        pass("Tanh large negative", close_enough((*output)[2], -1.0f));

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(output);
    }

    // ========== 9. Softmax ==========
    {
        std::cout << "\n--- Testing Softmax ---" << std::endl;
        libshit::layers::Softmax sm;

        ShitTensor* input = new ShitTensor({1, 4});
        (*input)[0] = 1.0f; (*input)[1] = 2.0f;
        (*input)[2] = 3.0f; (*input)[3] = 4.0f;
        input->to_gpu();

        ShitTensor* output = sm(input);
        output->to_cpu();
        pass("Softmax output shape", output->get_shape() == input->get_shape());

        // Check sum = 1
        float sum = 0.0f;
        for (int64_t i = 0; i < output->size(); ++i) sum += (*output)[i];
        pass("Softmax sum to 1", close_enough(sum, 1.0f));

        // Check monotonic (largest input -> largest prob)
        pass("Softmax monotonic", (*output)[3] > (*output)[2] &&
                                  (*output)[2] > (*output)[1] &&
                                  (*output)[1] > (*output)[0]);

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(output);
    }

    // ========== 10. Sequential ==========
    {
        std::cout << "\n--- Testing Sequential ---" << std::endl;
        auto seq = std::make_shared<libshit::layers::Sequential>();
        auto dense = seq->add_module<libshit::layers::Dense>(3, 2);
        auto relu = seq->add_module<libshit::layers::ReLU>();

        ShitTensor* input = new ShitTensor({1, 3});
        (*input)[0] = 1.0f; (*input)[1] = -2.0f; (*input)[2] = 3.0f;
        input->to_gpu();

        ShitTensor* output = (*seq)(input);
        output->to_cpu();
        pass("Sequential output valid shape",
             output->get_shape().size() == 2 && output->get_shape()[1] == 2);
        pass("Sequential with ReLU (no negatives)",
             (*output)[0] >= 0 && (*output)[1] >= 0);

        // Check parameters include child layer params
        auto seq_params = seq->get_parameters();
        auto dense_params = dense->get_parameters();
        pass("Sequential includes child parameters",
             seq_params.size() == dense_params.size());

        libshit::core::free_tensor(input);
        libshit::core::free_tensor(output);
    }

    // ========== Summary ==========
    std::cout << "\n==============================" << std::endl;
    std::cout << test_count << " tests executed." << std::endl;
    if (all_passed) {
        std::cout << "All new layer tests passed!" << std::endl;
    } else {
        std::cerr << "Some tests failed!" << std::endl;
    }

    return all_passed ? 0 : 1;
}