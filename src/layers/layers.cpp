#include "../include/libshit_core.h"
#include "../include/layers/layers.hpp"
#include <random>
#include <memory>
#include <algorithm>

namespace libshit::layers {
    libshit::core::ShitTensor* ShitModel::train_step(libshit::core::ShitTensor* input, libshit::core::ShitTensor* target) {
        auto& tape = libshit::core::ShitTape::instance();

        tape.start();

        const auto& output = (*this)(input);
        const auto& loss = loss_func(output, target);

        registry.clear();

        // fix the loss stuff with gradients
        auto loss_grad = registry.get_grad(loss);
        (*loss_grad)[0] = 1.0f;
        loss_grad->to_gpu();

        tape.backward(registry);
        tape.stop();

        // 4. Update Weights
        for (auto& p : get_parameters()) {
            auto grad = registry.get_grad(p.get());
            if (grad) {
                optimizer->step(*p, *grad);
            }
        }

        return loss;
    }

    void ShitModel::set_optimizer(std::unique_ptr<libshit::optim::ShitOptimizer> optimizer){
        this->optimizer = std::move(optimizer);
    }

    void ShitModel::train(std::vector<std::pair<libshit::core::ShitTensor*, libshit::core::ShitTensor*>>& data, int epochs) {
        // if u cant understand this then learn to code before 
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int batch_idx = 0;
            
            for (auto& [input, target] : data) {
                auto loss_tensor = this->train_step(input, target);
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
        bias = register_parameter(
            "bias",
            {1, out_features},
            libshit::core::ShitTensor::InitType::XavierUniform
        );
    }

    libshit::core::ShitTensor* Dense::call(libshit::core::ShitTensor* input) {
        auto x = libshit::core::matmul(input, weights.get());
        x = libshit::core::add(x, bias.get());
        return x;
    }

    // ReLU
    ReLU::ReLU() = default;

    void ReLU::build(libshit::core::ShitTensor* input) {
        // nun to train
        (void)input;
    }

    libshit::core::ShitTensor* ReLU::call(libshit::core::ShitTensor* input) {
        auto x = libshit::core::relu(input);
        return x;
    }

        // ========== LeakyReLU ==========

        void LeakyReLU::build(libshit::core::ShitTensor* input) {
            (void)input;
        }

        libshit::core::ShitTensor* LeakyReLU::call(libshit::core::ShitTensor* input) {
            return libshit::core::leaky_relu(input, alpha);
        }

        // ========== Sigmoid ==========

        void Sigmoid::build(libshit::core::ShitTensor* input) {
            (void)input;
        }

        libshit::core::ShitTensor* Sigmoid::call(libshit::core::ShitTensor* input) {
            return libshit::core::sigmoid(input);
        }

        // ========== Tanh ==========

        void Tanh::build(libshit::core::ShitTensor* input) {
            (void)input;
        }

        libshit::core::ShitTensor* Tanh::call(libshit::core::ShitTensor* input) {
            return libshit::core::tanh_act(input);
        }

        // ========== Softmax ==========

        void Softmax::build(libshit::core::ShitTensor* input) {
            (void)input;
        }

        libshit::core::ShitTensor* Softmax::call(libshit::core::ShitTensor* input) {
            return libshit::core::softmax(input);
        }

        // ========== Dropout ==========

        void Dropout::build(libshit::core::ShitTensor* input) {
            (void)input;
        }

        libshit::core::ShitTensor* Dropout::call(libshit::core::ShitTensor* input) {
            if (!training || probability == 0.0f) {
                // Identity in eval mode or zero prob
                return libshit::core::reshape(input, input->get_shape());
            }
            return libshit::core::dropout(input, probability);
        }

        // ========== Flatten ==========

        void Flatten::build(libshit::core::ShitTensor* input) {
            (void)input;
        }

        libshit::core::ShitTensor* Flatten::call(libshit::core::ShitTensor* input) {
            int64_t flattened = 1;
            for (auto d : input->get_shape()) flattened *= d;
            return libshit::core::reshape(input, {1, flattened});
        }

        // ========== Embedding ==========

        void Embedding::build(libshit::core::ShitTensor* input) {
            weight = register_parameter(
                "weight",
                {vocab_size, embedding_dim},
                libshit::core::ShitTensor::InitType::RandomUniform
            );
        }

        libshit::core::ShitTensor* Embedding::call(libshit::core::ShitTensor* input) {
            // Only scalar index supported for now
            if (input->size() != 1) {
                std::cerr << "LIBSHIT EMBEDDING ERROR: Only scalar index supported." << std::endl;
                exit(1);
            }
            input->to_cpu();
            int64_t idx = (int64_t)(*input)[0];
            return libshit::core::embedding_lookup(weight.get(), idx);
        }

        // ========== LayerNorm ==========

        void LayerNorm::build(libshit::core::ShitTensor* input) {
            gamma = register_parameter(
                "gamma",
                {normalized_shape},
                libshit::core::ShitTensor::InitType::None
            );
            beta = register_parameter(
                "beta",
                {normalized_shape},
                libshit::core::ShitTensor::InitType::None
            );
            // Initialize gamma to ones, beta to zeros
            gamma->to_cpu();
            beta->to_cpu();
            for (int64_t i = 0; i < normalized_shape; ++i) {
                (*gamma)[i] = 1.0f;
                (*beta)[i] = 0.0f;
            }
            gamma->to_gpu();
            beta->to_gpu();
        }

        libshit::core::ShitTensor* LayerNorm::call(libshit::core::ShitTensor* input) {
                    const auto& shape = input->get_shape();
                    if (shape.back() != normalized_shape) {
                        std::cerr << "LIBSHIT LAYERNORM ERROR: Last dim mismatch." << std::endl;
                        exit(1);
                    }

                    int64_t rows = 1;
                    for (size_t i = 0; i < shape.size() - 1; ++i) rows *= shape[i];
                    int64_t cols = shape.back();

                    auto reshaped = libshit::core::reshape(input, {rows, cols});

                    // mean = sum_all / (rows * cols) — simplified: uses global mean
                    // Ideally per-row; this works for single-sample batches
                    auto mean_t = libshit::core::sum_all(reshaped);
                    mean_t->to_cpu();
                    float mean_val = mean_t->cpu_ptr()[0] / (float)cols;
                    libshit::core::free_tensor(mean_t);

                    // x - mean
                    auto mean_tensor = new libshit::core::ShitTensor({1, cols});
                    for (int64_t i = 0; i < cols; ++i) (*mean_tensor)[i] = mean_val;
                    mean_tensor->to_gpu();
                    auto centered = libshit::core::subtract(reshaped, mean_tensor);
                    libshit::core::free_tensor(mean_tensor);

                    // variance
                    auto squared = libshit::core::pow_op(centered, 2.0f);
                    auto var_t = libshit::core::sum_all(squared);
                    var_t->to_cpu();
                    float var_val = var_t->cpu_ptr()[0] / (float)cols;
                    float std_val = std::sqrt(var_val + eps);
                    libshit::core::free_tensor(var_t);

                    // normalize
                    auto normalized = libshit::core::div_scalar(centered, std_val);
                    libshit::core::free_tensor(centered);
                    libshit::core::free_tensor(squared);

                    // gamma and beta broadcast: reshape to {1, cols}
                    auto gamma_2d = libshit::core::reshape(gamma.get(), {1, cols});
                    auto beta_2d = libshit::core::reshape(beta.get(), {1, cols});

                    auto scaled = libshit::core::multiply(normalized, gamma_2d);
                    libshit::core::free_tensor(normalized);
                    libshit::core::free_tensor(gamma_2d);

                    auto result = libshit::core::add(scaled, beta_2d);
                    libshit::core::free_tensor(scaled);
                    libshit::core::free_tensor(beta_2d);

                    // Reshape back
                    auto final_out = libshit::core::reshape(result, shape);
                    libshit::core::free_tensor(result);
                    return final_out;
                }

        // ========== RMSNorm ==========

        void RMSNorm::build(libshit::core::ShitTensor* input) {
            gamma = register_parameter(
                "gamma",
                {normalized_shape},
                libshit::core::ShitTensor::InitType::None
            );
            // Initialize gamma to ones
            gamma->to_cpu();
            for (int64_t i = 0; i < normalized_shape; ++i) {
                (*gamma)[i] = 1.0f;
            }
            gamma->to_gpu();
        }

        libshit::core::ShitTensor* RMSNorm::call(libshit::core::ShitTensor* input) {
            const auto& shape = input->get_shape();
            if (shape.back() != normalized_shape) {
                std::cerr << "LIBSHIT RMSNORM ERROR: Last dim mismatch." << std::endl;
                exit(1);
            }

            // x / sqrt(mean(x^2) + eps) * gamma  (broadcast gamma over leading dims)
            int64_t rows = 1;
            for (size_t i = 0; i < shape.size() - 1; ++i) rows *= shape[i];
            int64_t cols = shape.back();

            auto reshaped = libshit::core::reshape(input, {rows, cols});

            // Compute row-wise RMS: mean of squares per row
            // For simplicity, do scalar RMS over the whole tensor (works for single sample)
            auto squared = libshit::core::pow_op(reshaped, 2.0f);
            auto sum_sq = libshit::core::sum_all(squared);
            sum_sq->to_cpu();
            float rms_val = std::sqrt(sum_sq->cpu_ptr()[0] / (float)(rows * cols) + eps);
            libshit::core::free_tensor(squared);
            libshit::core::free_tensor(sum_sq);

            // Normalize
            auto normalized = libshit::core::div_scalar(reshaped, rms_val);
            libshit::core::free_tensor(reshaped);

            // Reshape gamma to broadcast: {1, cols}
            auto gamma_2d = libshit::core::reshape(gamma.get(), {1, cols});

            // Element-wise multiply with gamma
            auto scaled = libshit::core::multiply(normalized, gamma_2d);
            libshit::core::free_tensor(normalized);
            libshit::core::free_tensor(gamma_2d);

            // Reshape back
            auto result = libshit::core::reshape(scaled, shape);
            libshit::core::free_tensor(scaled);
            return result;
        }

        // ========== Sequential ==========

        void Sequential::build(libshit::core::ShitTensor* input) {
            // Build all child layers
            for (auto& layer : get_layers()) {
                (*layer)(input);
            }
        }

        libshit::core::ShitTensor* Sequential::call(libshit::core::ShitTensor* input) {
            auto x = input;
            std::vector<libshit::core::ShitTensor*> temps;

            for (auto& layer : get_layers()) {
                auto out = (*layer)(x);
                if (x != input) temps.push_back(x);
                x = out;
            }

            // Cleanup intermediate tensors
            for (auto t : temps) libshit::core::free_tensor(t);
            return x;
        }
    }