#include "../include/save/save.hpp"

namespace libshit::save {
    void ShitSerializer::save(libshit::layers::ShitModel& model, const std::string& path) {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write("SHIT", 4);
        uint32_t version = LIBSHIT_SERIALIZER_VERSION;
        ofs.write((char*)&version, sizeof(version));

        // get params and save
        auto params = model.get_parameters();
        for (auto& p : params) {
            save_tensor(ofs, p.get());
        }
    }

    void ShitSerializer::load(libshit::layers::ShitModel& model, const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        char header[4];
        ifs.read(header, 4);

        if (std::string(header, 4) != "SHIT") {
            throw std::runtime_error("Invalid file format: Not a ShitModel file.");
        }

        uint32_t version;
        ifs.read((char*)&version, sizeof(version));
        if(version != LIBSHIT_SERIALIZER_VERSION){
            throw std::runtime_error("Unsupported libshit version: file was saved with libshit version \"" + std::to_string(version) + "\" while using serializer version \"" + std::to_string(LIBSHIT_SERIALIZER_VERSION) + "\"");
        }

        auto params = model.get_parameters();
        for (auto& p : params) {
            // calculate size again
            size_t num_elements = 1;
            for (auto dim : p->get_shape()) num_elements *= dim;
            
            // load the data right into the tensor
            ifs.read((char*)p->cpu_ptr(), num_elements * sizeof(float));
            
            // put it on the gpu
            p->to_gpu(); 
        }
    }

    void ShitSerializer::save_tensor(std::ofstream& ofs, libshit::core::ShitTensor* t) {
        // get the size of the tensor
        size_t num_elements = 1;
        auto shape = t->get_shape(); 
        for (int64_t dim : shape) num_elements *= dim;
        size_t total_bytes = num_elements * sizeof(float);

        // send back to cpu
        t->to_cpu(); 
        
        // write that shi
        ofs.write((char*)t->cpu_ptr(), total_bytes);
    }
}