#ifndef LIBSHIT_LAYER_H
#define LIBSHIT_LAYER_H
#include "../include/libshit_core.h"
#include <unordered_map>

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
        libshit::core::ShitTensor* train_step(libshit::core::ShitTensor* input, libshit::core::ShitTensor* target, float lr);
    
    public:
        void train(std::vector<std::pair<libshit::core::ShitTensor*, libshit::core::ShitTensor*>>& data, int epochs, float lr);
    };



    // Dense/Linear Layer just a normal layer 
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
}

#endif