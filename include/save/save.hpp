#ifndef LIBSHIT_SAVE_H
#define LIBSHIT_SAVE_H

#define LIBSHIT_SERIALIZER_VERSION 1

#include <fstream>
#include <string>
#include "../include/libshit_core.h"
#include "../include/layers/layers.hpp"

namespace libshit::save {
    class SHIT_API ShitSerializer {
    public:
        static void save(libshit::layers::ShitModel& model, const std::string& path);
        static void load(libshit::layers::ShitModel& model, const std::string& path);

    private:
        static void save_tensor(std::ofstream& ofs, libshit::core::ShitTensor* t);
    };
}

#endif