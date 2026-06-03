#include "../include/optim/optimizer.hpp"

void libshit::optim::SGD::step(libshit::core::ShitTensor& param, libshit::core::ShitTensor& grad){
    sgd(&param, &grad, lr);
}