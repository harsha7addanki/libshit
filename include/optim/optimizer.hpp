#ifndef LIBSHIT_OPTIMIZER_H
#define LIBSHIT_OPTIMIZER_H
#include "libshit_core.h"

namespace libshit::optim {
    class SHIT_API ShitOptimizer {
    public:
        virtual ~ShitOptimizer() = default;
        // The core interface: what every optimizer MUST do
        virtual void step(libshit::core::ShitTensor& param, libshit::core::ShitTensor& grad) = 0;
    };

    class SHIT_API SGD : public ShitOptimizer {
    float lr;
    public:
        SGD(float learning_rate) : lr(learning_rate) {}
        void step(libshit::core::ShitTensor& param, libshit::core::ShitTensor& grad) override;
    };
    
    SHIT_API void sgd(libshit::core::ShitTensor* weights, libshit::core::ShitTensor* grad_weights, float learning_rate);
}
#endif