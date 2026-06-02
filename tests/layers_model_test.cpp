#include "../include/layers/layers.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

class SimpleRegressionModel : public libshit::layers::ShitModel {
public:
    std::shared_ptr<libshit::layers::Dense> fc;

    void build(libshit::core::ShitTensor* input) override {
        fc = add<libshit::layers::Dense>(2, 1);
    }

    libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override {
        return (*fc)(input);
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

    model.train(dataset, 20, 0.01f);

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
    return 1;
}
