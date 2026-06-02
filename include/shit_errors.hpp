#ifndef SHIT_ERRORS_H
#define SHIT_ERRORS_H

#include <iostream>
#include <cuda_runtime.h>

// very basic error check for cuda errors
#define SHIT_CHECK(ans) { libshit::core::cuda_assert((ans), __FILE__, __LINE__); }
namespace libshit::core{
    inline void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true) {
        if (code != cudaSuccess) {
            std::cerr << "\n=========== LIBSHIT CRITICAL ERROR =========== " << std::endl;
            std::cerr << "GPU Error Code: " << cudaGetErrorString(code) << std::endl;
            std::cerr << "Location:       " << file << ":" << line << std::endl;
            std::cerr << "===============================================\n" << std::endl;
            
            if (abort) {
                // reset cuda so we dont fuck it up
                cudaDeviceReset();
                exit(code);
            }
        }
    }
}

#endif
