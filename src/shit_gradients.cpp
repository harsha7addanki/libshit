#include "shit_gradients.hpp"
#include "shit_errors.hpp"
#include "shit_tensor.hpp"

namespace libshit::core {
    void ShitTape::backward(ShitGradientRegistry& registry) {
        for (auto node = nodes.rbegin(); node != nodes.rend(); ++node) {
            (*node)->backward(registry);
        }
    }

    ShitTensor* ShitGradientRegistry::get_grad(ShitTensor* t){
        if (grad_map.find(t) == grad_map.end()) {
            auto t_shape = t->get_shape();
            ShitTensor* new_grad = new ShitTensor(t_shape);
            
            cudaMemset(new_grad->gpu_ptr(), 0, t->size() * sizeof(float));

            grad_map[t] = new_grad;
        }
        
        return grad_map[t];
    }

    void ShitGradientRegistry::clear() {
        for (auto& [key, tensor] : grad_map) {
            delete tensor;
        }
        grad_map.clear();
    }

    ShitGradientRegistry::~ShitGradientRegistry() {
        clear();
    }


    // MatMulNode implementation
    void MatMulNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_A = registry.get_grad(A);
        ShitTensor* grad_B = registry.get_grad(B);
        ShitTensor* grad_out = registry.get_grad(out);

        backward_matmul(grad_out, A, B, grad_A, grad_B);
    }

    void MatMulNode::forward() {
        out = matmul(A, B);
    }


    // AddNode implementation
    void AddNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_A = registry.get_grad(A);
        ShitTensor* grad_B = registry.get_grad(B);
        ShitTensor* grad_out = registry.get_grad(out);

        backward_add(grad_out, A, B, grad_A, grad_B);
    }

    void AddNode::forward() {
        out = add(A, B);
    }


    // ReLUNode implementation
    void ReLUNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_input = registry.get_grad(input);
        ShitTensor* grad_output = registry.get_grad(out);

        backward_relu(grad_output, input, grad_input);
    }

    void ReLUNode::forward() {
        out = relu(input);
    }


    // MSENode implementation
    void MSENode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_input = registry.get_grad(input);
        ShitTensor* grad_output = registry.get_grad(out);

        backward_mse(grad_output, input, target, grad_input);
    }

    void MSENode::forward() {
        out = mse(input, target);
    }


    // SubtractNode implementation
    void SubtractNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_A = registry.get_grad(A);
        ShitTensor* grad_B = registry.get_grad(B);
        ShitTensor* grad_out = registry.get_grad(out);
        backward_subtract(grad_out, A, B, grad_A, grad_B);
    }

    void SubtractNode::forward() {
        out = subtract(A, B);
    }


    // MultiplyNode implementation
    void MultiplyNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_A = registry.get_grad(A);
        ShitTensor* grad_B = registry.get_grad(B);
        ShitTensor* grad_out = registry.get_grad(out);
        backward_multiply(grad_out, A, B, grad_A, grad_B);
    }

    void MultiplyNode::forward() {
        out = multiply(A, B);
    }


    // NegateNode implementation
    void NegateNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_input = registry.get_grad(input);
        ShitTensor* grad_out = registry.get_grad(out);
        backward_negate(grad_out, input, grad_input);
    }

    void NegateNode::forward() {
        out = negate(input);
    }


    // SigmoidNode implementation
    void SigmoidNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_input = registry.get_grad(input);
        ShitTensor* grad_out = registry.get_grad(out);
        backward_sigmoid(grad_out, out, grad_input);
    }

    void SigmoidNode::forward() {
        out = sigmoid(input);
    }


    // TanhNode implementation
    void TanhNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_input = registry.get_grad(input);
        ShitTensor* grad_out = registry.get_grad(out);
        backward_tanh(grad_out, out, grad_input);
    }

    void TanhNode::forward() {
        out = tanh_act(input);
    }


    // PowNode implementation
    void PowNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_A = registry.get_grad(A);
        ShitTensor* grad_out = registry.get_grad(out);
        backward_pow(grad_out, A, exponent, grad_A);
    }

    void PowNode::forward() {
        out = pow_op(A, exponent);
    }


    // SumAllNode implementation
    void SumAllNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_input = registry.get_grad(input);
        ShitTensor* grad_out = registry.get_grad(out);
        backward_sum_all(grad_out, input, grad_input);
    }

    void SumAllNode::forward() {
        out = sum_all(input);
    }


    // TransposeNode implementation
    void TransposeNode::backward(ShitGradientRegistry& registry) {
        ShitTensor* grad_input = registry.get_grad(input);
        ShitTensor* grad_out = registry.get_grad(out);
        backward_transpose(grad_out, input, grad_input);
    }

    void TransposeNode::forward() {
        out = transpose(input);
    }
}