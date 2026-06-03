#include "../include/layers/layers.hpp"
#include "../include/optim/optimizer.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <string>

static void print_tensor(const libshit::core::ShitTensor* tensor, const std::string& name) {
    auto mutable_tensor = const_cast<libshit::core::ShitTensor*>(tensor);
    mutable_tensor->to_cpu();
    const auto& shape = mutable_tensor->get_shape();

    std::cout << name << " shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << "] values=[";

    for (int64_t i = 0; i < mutable_tensor->size(); ++i) {
        std::cout << mutable_tensor->cpu_ptr()[i];
        if (i + 1 < mutable_tensor->size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

class SimpleRegressionModel : public libshit::layers::ShitModel {
public:
    std::shared_ptr<libshit::layers::Dense> fc;
    std::shared_ptr<libshit::layers::ReLU> relu;

    void build(libshit::core::ShitTensor* input) override {
        fc = add<libshit::layers::Dense>(2, 1);
        relu = add<libshit::layers::ReLU>();
    }

    libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override {
        auto x = (*fc)(input);
        x = (*relu)(x);
        return x;
    }
};

static float compute_loss(libshit::core::ShitTensor* prediction, libshit::core::ShitTensor* target) {
    auto loss = libshit::core::mse(prediction, target);
    loss->to_cpu();
    float value = loss->cpu_ptr()[0];
    libshit::core::free_tensor(loss);
    return value;
}

int main() {
    using namespace libshit::core;

    std::cout << "Starting layers API model training test..." << std::endl;

    SimpleRegressionModel model;
    auto optimizer = std::make_unique<libshit::optim::SGD>(0.01f);
    model.set_optimizer(std::move(optimizer));
    model.set_loss(libshit::core::mse);

    std::vector<std::pair<ShitTensor*, ShitTensor*>> dataset;
    {
        auto x1 = new ShitTensor({1, 2});
        (*x1)[0] = 1.0f; (*x1)[1] = 1.0f;
        x1->to_gpu();
        auto y1 = new ShitTensor({1});
        (*y1)[0] = 2.0f;
        y1->to_gpu();
        dataset.emplace_back(x1, y1);

        auto x2 = new ShitTensor({1, 2});
        (*x2)[0] = 2.0f; (*x2)[1] = 0.0f;
        x2->to_gpu();
        auto y2 = new ShitTensor({1});
        (*y2)[0] = 2.0f;
        y2->to_gpu();
        dataset.emplace_back(x2, y2);

        auto x3 = new ShitTensor({1, 2});
        (*x3)[0] = 3.0f; (*x3)[1] = 1.0f;
        x3->to_gpu();
        auto y3 = new ShitTensor({1});
        (*y3)[0] = 4.0f;
        y3->to_gpu();
        dataset.emplace_back(x3, y3);
    }

    // Evaluate initial loss on the first sample.
    auto baseline_output = model(dataset[0].first);
    float baseline_loss = compute_loss(baseline_output, dataset[0].second);
    libshit::core::free_tensor(baseline_output);

    std::cout << "Baseline loss: " << baseline_loss << std::endl;

    model.train(dataset, 40);

    auto final_output = model(dataset[0].first);
    float final_loss = compute_loss(final_output, dataset[0].second);
    libshit::core::free_tensor(final_output);

    std::cout << "Final loss: " << final_loss << std::endl;

    for (auto& [input, target] : dataset) {
        libshit::core::free_tensor(input);
        libshit::core::free_tensor(target);
    }

    if (final_loss < baseline_loss) {
        std::cout << "Layers model training test passed." << std::endl;
        return 0;
    }

    std::cerr << "Layers model training test failed: loss did not decrease." << std::endl;
    std::cerr << "Final parameter values:" << std::endl;
    auto params = model.fc->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        const std::string name = (i == 0) ? "weights" : (i == 1) ? "bias" : ("param" + std::to_string(i));
        print_tensor(params[i].get(), name);
    }
    return 1;
}
