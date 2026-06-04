#ifndef LIBSHIT_LAYER_H
#define LIBSHIT_LAYER_H
#include "../include/libshit_core.h"
#include "../include/optim/optimizer.hpp"
#include <unordered_map>
#include <functional>
#include <random>

// Api to easily create DNN based AI models
namespace libshit::layers {
    class SHIT_API ShitLayer {
    protected:
        bool training = true;
        bool built = false;

        std::shared_ptr<libshit::core::ShitTensor> register_parameter(const std::string& name, std::vector<int64_t> shape, libshit::core::ShitTensor::InitType init = libshit::core::ShitTensor::InitType::None){
            auto param = std::make_shared<libshit::core::ShitTensor>(shape, init);
            param->to_gpu();
            parameters[name] = param;
            return param;
        }
    private:
        std::unordered_map<std::string, std::shared_ptr<libshit::core::ShitTensor>> parameters;
        std::vector<std::shared_ptr<ShitLayer>> layers;
    public:
        virtual libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) = 0;
        virtual void build(libshit::core::ShitTensor* input) = 0; // pure virtual for building the layer (initializing parameters)
        virtual ~ShitLayer() = default;
        
        libshit::core::ShitTensor* operator()(libshit::core::ShitTensor* input) {
            if (!built) {
                build(input);
                built = true;
            }
            return call(input);
        }

        template<typename T, typename... Args>
        std::shared_ptr<T> add(Args&&... args) {
            auto layer = std::make_shared<T>(std::forward<Args>(args)...);
            this->layers.push_back(layer);
            return layer;
        }

        virtual std::vector<std::shared_ptr<libshit::core::ShitTensor>> get_parameters() {
            std::vector<std::shared_ptr<libshit::core::ShitTensor>> params;
            
            // get our own parameters
            for (auto const& [name, tensor] : parameters) {
                params.push_back(tensor);
            }
            
            // get any child layer parameters
            for (auto& child : layers) {
                // by using their get_parameters method we go through the whole tree
                auto child_params = child->get_parameters();
                params.insert(params.end(), child_params.begin(), child_params.end());
            }
            return params;
        }

        std::vector<std::shared_ptr<ShitLayer>> get_layers(){ return layers; }

        void train() { training = true; }
        void eval() { training = false; }
    };

    class SHIT_API ShitModel: public ShitLayer {
    protected:
        libshit::core::ShitGradientRegistry registry;
        std::unique_ptr<libshit::optim::ShitOptimizer> optimizer;
        std::function<libshit::core::ShitTensor*(libshit::core::ShitTensor*, libshit::core::ShitTensor*)> loss_func;
        libshit::core::ShitTensor* train_step(libshit::core::ShitTensor* input, libshit::core::ShitTensor* target);
    
    public:
        void set_optimizer(std::unique_ptr<libshit::optim::ShitOptimizer> optimizer);
        void set_loss(std::function<libshit::core::ShitTensor*(libshit::core::ShitTensor*, libshit::core::ShitTensor*)> loss){loss_func = loss;};
        void train(std::vector<std::pair<libshit::core::ShitTensor*, libshit::core::ShitTensor*>>& data, int epochs);
    };



    // tf/keras on top fuck torch
    // Dense Layer just a normal layer 
    class SHIT_API Dense : public ShitLayer {
    protected:
        int in_features;
        int out_features;

        std::shared_ptr<libshit::core::ShitTensor> weights;
        std::shared_ptr<libshit::core::ShitTensor> bias;

    public:
        Dense(int in, int out): in_features(in), out_features(out) {}
        void build(libshit::core::ShitTensor* input) override;
        libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
    };

    // ReLU layer
    class SHIT_API ReLU : public ShitLayer {
    public:
        ReLU();
        void build(libshit::core::ShitTensor* input) override;
        libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
    };

        // ========== New Layers ==========

        // LeakyReLU
        class SHIT_API LeakyReLU : public ShitLayer {
        protected:
            float alpha;
        public:
            LeakyReLU(float alpha_val = 0.01f) : alpha(alpha_val) {}
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
        };

        // Sigmoid
        class SHIT_API Sigmoid : public ShitLayer {
        public:
            Sigmoid() = default;
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
        };

        // Tanh
        class SHIT_API Tanh : public ShitLayer {
        public:
            Tanh() = default;
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
        };

        // Softmax
        class SHIT_API Softmax : public ShitLayer {
        public:
            Softmax() = default;
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
        };

        // Dropout
        class SHIT_API Dropout : public ShitLayer {
        protected:
            float probability;
        public:
            Dropout(float p = 0.5f) : probability(p) {}
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
        };

        // Flatten
        class SHIT_API Flatten : public ShitLayer {
        public:
            Flatten() = default;
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
        };

        // Embedding
        class SHIT_API Embedding : public ShitLayer {
        protected:
            int64_t vocab_size;
            int64_t embedding_dim;
            std::shared_ptr<libshit::core::ShitTensor> weight;
        public:
            Embedding(int64_t vocab, int64_t dim) : vocab_size(vocab), embedding_dim(dim) {}
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
            int64_t get_embedding_dim() const { return embedding_dim; }
        };

        // LayerNorm
        class SHIT_API LayerNorm : public ShitLayer {
        protected:
            int64_t normalized_shape;
            float eps;
            std::shared_ptr<libshit::core::ShitTensor> gamma;
            std::shared_ptr<libshit::core::ShitTensor> beta;
        public:
            LayerNorm(int64_t norm_shape, float epsilon = 1e-5f)
                : normalized_shape(norm_shape), eps(epsilon) {}
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
        };

        // RMSNorm
        class SHIT_API RMSNorm : public ShitLayer {
        protected:
            int64_t normalized_shape;
            float eps;
            std::shared_ptr<libshit::core::ShitTensor> gamma;
        public:
            RMSNorm(int64_t norm_shape, float epsilon = 1e-5f)
                : normalized_shape(norm_shape), eps(epsilon) {}
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;
        };

        // Sequential container
        class SHIT_API Sequential : public ShitLayer {
        public:
            Sequential() = default;
            void build(libshit::core::ShitTensor* input) override;
            libshit::core::ShitTensor* call(libshit::core::ShitTensor* input) override;

            template<typename T, typename... Args>
            std::shared_ptr<T> add_module(Args&&... args) {
                return add<T>(std::forward<Args>(args)...);
            }
        };
    }

#endif