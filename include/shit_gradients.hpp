#ifndef SHIT_GRADIENTS_H
#define SHIT_GRADIENTS_H

#ifdef _WIN32
    #ifdef LIBSHIT_EXPORTS
        #define SHIT_API __declspec(dllexport)
    #else
        #define SHIT_API __declspec(dllimport)
    #endif
#else
    #define SHIT_API
#endif

#include "shit_tensor.hpp"
#include <vector>
#include <unordered_map>
#include <iostream>

namespace libshit::core{
    
    class SHIT_API ShitGradientRegistry{
        public:
        std::unordered_map<ShitTensor*, ShitTensor*> grad_map;
        ShitTensor* get_grad(ShitTensor* t);
        void clear();
        ~ShitGradientRegistry();
    };

    struct SHIT_API ShitOperatorNode {
        virtual void forward() = 0;
        virtual void backward(ShitGradientRegistry& registry) = 0;
        virtual ~ShitOperatorNode() = default;
    };

    class SHIT_API ShitTape{
    private:
        std::vector<std::unique_ptr<ShitOperatorNode>> nodes;
        bool active = false;
        ShitTape() = default;
    public:
       static ShitTape& instance() { 
            static ShitTape t; 
            return t; 
        }
        
        ShitTape(const ShitTape&) = delete; 
        ShitTape& operator=(const ShitTape&) = delete;
        
        void start() { active = true; nodes.clear(); }
        void stop()  { active = false; }
        bool is_active() { return active; }

        void push_node(std::unique_ptr<ShitOperatorNode> node) { nodes.push_back(std::move(node)); }
        std::vector<std::unique_ptr<ShitOperatorNode>>& get_nodes() { return nodes; }

        void backward(ShitGradientRegistry& registry);

    };



    struct SHIT_API MatMulNode : public ShitOperatorNode {
        ShitTensor *A, *B, *out;
        MatMulNode(ShitTensor *a, ShitTensor *b, ShitTensor *o) : A(a), B(b), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API AddNode : public ShitOperatorNode {
        ShitTensor *A, *B, *out;
        AddNode(ShitTensor *a, ShitTensor *b, ShitTensor *o) : A(a), B(b), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API ReLUNode : public ShitOperatorNode {
        ShitTensor *input, *out;
        ReLUNode(ShitTensor *in, ShitTensor *out) : input(in), out(out) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API MSENode : public ShitOperatorNode {
        ShitTensor *input, *target, *out;
        MSENode(ShitTensor *in, ShitTensor *tar, ShitTensor *out) : input(in), target(tar), out(out) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    // New graph nodes
    struct SHIT_API SubtractNode : public ShitOperatorNode {
        ShitTensor *A, *B, *out;
        SubtractNode(ShitTensor *a, ShitTensor *b, ShitTensor *o) : A(a), B(b), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API MultiplyNode : public ShitOperatorNode {
        ShitTensor *A, *B, *out;
        MultiplyNode(ShitTensor *a, ShitTensor *b, ShitTensor *o) : A(a), B(b), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API NegateNode : public ShitOperatorNode {
        ShitTensor *input, *out;
        NegateNode(ShitTensor *in, ShitTensor *o) : input(in), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API SigmoidNode : public ShitOperatorNode {
        ShitTensor *input, *out;
        SigmoidNode(ShitTensor *in, ShitTensor *o) : input(in), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API TanhNode : public ShitOperatorNode {
        ShitTensor *input, *out;
        TanhNode(ShitTensor *in, ShitTensor *o) : input(in), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API PowNode : public ShitOperatorNode {
        ShitTensor *A, *out;
        float exponent;
        PowNode(ShitTensor *a, float exp, ShitTensor *o) : A(a), exponent(exp), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API SumAllNode : public ShitOperatorNode {
        ShitTensor *input, *out;
        SumAllNode(ShitTensor *in, ShitTensor *o) : input(in), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    struct SHIT_API TransposeNode : public ShitOperatorNode {
        ShitTensor *input, *out;
        TransposeNode(ShitTensor *in, ShitTensor *o) : input(in), out(o) {}
        void backward(ShitGradientRegistry& registry) override;
        void forward() override;
    };

    // Existing backward declarations
    SHIT_API void backward_add(ShitTensor* grad_out, ShitTensor* A, ShitTensor* B, ShitTensor* grad_A, ShitTensor* grad_B);
    SHIT_API void backward_matmul(ShitTensor* grad_out, ShitTensor* A, ShitTensor* B, ShitTensor* grad_A, ShitTensor* grad_B);
    SHIT_API void backward_relu(ShitTensor* grad_out, ShitTensor* input, ShitTensor* grad_input);
    SHIT_API void backward_mse(ShitTensor* grad_out, ShitTensor* pred, ShitTensor* target, ShitTensor* grad_pred);

    // New backward declarations
    SHIT_API void backward_subtract(ShitTensor* grad_out, ShitTensor* A, ShitTensor* B, ShitTensor* grad_A, ShitTensor* grad_B);
    SHIT_API void backward_multiply(ShitTensor* grad_out, ShitTensor* A, ShitTensor* B, ShitTensor* grad_A, ShitTensor* grad_B);
    SHIT_API void backward_negate(ShitTensor* grad_out, ShitTensor* input, ShitTensor* grad_input);
    SHIT_API void backward_sigmoid(ShitTensor* grad_out, ShitTensor* output, ShitTensor* grad_input);
    SHIT_API void backward_tanh(ShitTensor* grad_out, ShitTensor* output, ShitTensor* grad_input);
    SHIT_API void backward_pow(ShitTensor* grad_out, ShitTensor* A, float exponent, ShitTensor* grad_A);
    SHIT_API void backward_sum_all(ShitTensor* grad_out, ShitTensor* A, ShitTensor* grad_A);
    SHIT_API void backward_transpose(ShitTensor* grad_out, ShitTensor* A, ShitTensor* grad_A);

    SHIT_API void update_weights(ShitTensor* weights, ShitTensor* grad_weights, float learning_rate);
}

#endif