#include "../include/libshit_core.h"
#include "../include/layers/layers.hpp"
#include <random>

namespace libshit::layers {
    libshit::core::ShitTensor* ShitModel::train_step(libshit::core::ShitTensor* input, libshit::core::ShitTensor* target, float lr) {
        auto& tape = libshit::core::ShitTape::instance();

        tape.start();

        const auto& output = (*this)(input);
        const auto& loss = libshit::core::mse(output, target);

        registry.clear();

        // Seed the loss gradient so backward propagation happens.
        auto loss_grad = registry.get_grad(loss);
        (*loss_grad)[0] = 1.0f;
        loss_grad->to_gpu();

        tape.backward(registry);
        tape.stop();

        // 4. Update Weights
        for (auto& p : get_parameters()) {
            auto grad = registry.get_grad(p.get());
            if (grad) {
                libshit::core::update_weights(p.get(), grad, lr);
            }
        }

        return loss;
    }

    void ShitModel::train(std::vector<std::pair<libshit::core::ShitTensor*, libshit::core::ShitTensor*>>& data, int epochs, float lr) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int batch_idx = 0;
            
            for (auto& [input, target] : data) {
                auto loss_tensor = this->train_step(input, target, lr);
                loss_tensor->to_cpu();
                float current_loss = loss_tensor->cpu_ptr()[0];

                epoch_loss += current_loss;
            
                if (batch_idx % 10 == 0) {
                    std::cout << "Epoch [" << epoch << "] Batch [" << batch_idx 
                            << "] Loss: " << current_loss << std::endl;
                }
                batch_idx++;
            }
            
            std::cout << "--- End of Epoch " << epoch << " | Avg Loss: " 
                  << (epoch_loss / data.size()) << " ---" << std::endl;
        }
    }
    
    // Dense Layer
    void Dense::build(libshit::core::ShitTensor* input) {
        weights = register_parameter(
            "weights",
            {in_features, out_features},
            libshit::core::ShitTensor::InitType::XavierUniform
        );
        bias = register_parameter("bias", {1, out_features});
    }

    libshit::core::ShitTensor* Dense::call(libshit::core::ShitTensor* input) {
        auto x = libshit::core::matmul(input, weights.get());
        x = libshit::core::add(x, bias.get());
        return x;
    }

    // ReLU
    ReLU::ReLU() = default;

    void ReLU::build(libshit::core::ShitTensor* input) {
        // ReLU has no trainable parameters.
        (void)input;
    }

    libshit::core::ShitTensor* ReLU::call(libshit::core::ShitTensor* input) {
        auto x = libshit::core::relu(input);
        return x;
    }
}