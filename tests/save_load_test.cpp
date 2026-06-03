#include "../include/layers/layers.hpp"
#include "../include/save/save.hpp"
#include <iostream>
#include <cassert>

class SaveLoadModel : public libshit::layers::ShitModel {
public:
    std::shared_ptr<libshit::layers::Dense> fc;
    void build(libshit::core::ShitTensor* input) override {
        fc = add<libshit::layers::Dense>(2, 1);
    }
    libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override {
        return (*fc)(input);
    }
};

int main() {
    using namespace libshit::core;
    
    // 1. Setup Data
    ShitTensor* input = new ShitTensor({1, 2});
    (*input)[0] = 0.5f; (*input)[1] = -1.5f;
    input->to_gpu();

    SaveLoadModel model_a, model_b;
    model_a.build(input);
    model_b.build(input);

    // 2. Initialize Model A with known weights
    auto pA = model_a.get_parameters();
    for(auto& p : pA) {
        p->to_cpu();
        float* data = p->cpu_ptr();
        for(int64_t i = 0; i < p->size(); ++i) data[i] = 0.5f; // Set all to 0.5 for test
        p->to_gpu();
    }

    // 3. Save
    const std::string path = "test_run.shit";
    libshit::save::ShitSerializer::save(model_a, path);

    // 4. Load into Model B
    // CRITICAL: Ensure sizes match before loading to avoid vector out-of-range
    auto pB = model_b.get_parameters();
    if(pA.size() != pB.size()) {
        std::cerr << "Architecture mismatch! A: " << pA.size() << " B: " << pB.size() << std::endl;
        return 1;
    }
    
    libshit::save::ShitSerializer::load(model_b, path);

    // 5. Verification
    for (size_t i = 0; i < pA.size(); ++i) {
        pA[i]->to_cpu(); pB[i]->to_cpu();
        if(pA[i]->size() != pB[i]->size()) {
            std::cerr << "Size mismatch at index " << i << std::endl;
            return 1;
        }
        for (int64_t j = 0; j < pA[i]->size(); ++j) {
            if (std::abs(pA[i]->cpu_ptr()[j] - pB[i]->cpu_ptr()[j]) > 1e-5f) {
                std::cerr << "Data mismatch at [" << i << "][" << j << "]" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Successfully saved and loaded identical weights." << std::endl;
    
    // Cleanup
    free_tensor(input);
    return 0;
}