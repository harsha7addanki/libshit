cmake_minimum_required(VERSION 3.18)
project(libshit LANGUAGES CXX CUDA)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# THIS IS THE FIX: Explicitly find the CUDA toolkit components
find_package(CUDAToolkit REQUIRED)

# 1. Force modern C++ standards for both CPU and GPU code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. Tell CMake to automatically detect your RTX GPU architecture
set(CMAKE_CUDA_ARCHITECTURES native)

# 3. Define the library and bundle all source files together
add_library(libshit SHARED
    src/shit_tensor.cpp
    src/shit_kernels.cu
    src/shit_gradients.cpp
    src/shit_gradients.cu
    src/layers/layers.cpp
    include/libshit_core.h
    include/layers/layers.hpp
    include/shit_gradients.hpp
    include/shit_errors.hpp
    include/shit_tensor.hpp
)

target_compile_definitions(libshit PRIVATE LIBSHIT_EXPORTS)

# 4. Use modern target-based include lookups
#    Make sure CUDA headers are available to any consumer of our public headers
#    that include CUDA runtime types such as cuda_runtime.h.
target_include_directories(libshit PUBLIC include
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)

# 5. Link the CUDA runtime library (Now fully visible thanks to find_package)
target_link_libraries(libshit PRIVATE CUDA::cudart)

# 6. Inject performance optimizations ONLY during Release builds
target_compile_options(libshit PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math -O3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
)

# 7. Add device-side CUDA debug symbols during Debug builds
target_compile_options(libshit PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -g>
)


# This is where libshit gets to be tested
add_executable(basic_tests
    tests/main_test.cpp
)

add_executable(grad_tests
    tests/grad_test.cpp
)

add_executable(grad_loss_scaling_tests
    tests/grad_loss_scaling_test.cpp
)

add_executable(layers_model_tests
    tests/layers_model_test.cpp
)

add_executable(mse_tests
    tests/mse_test.cpp
)

target_include_directories(grad_tests PRIVATE include
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(grad_tests PRIVATE libshit)

target_include_directories(grad_loss_scaling_tests PRIVATE include
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(grad_loss_scaling_tests PRIVATE libshit)

target_include_directories(layers_model_tests PRIVATE include
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(layers_model_tests PRIVATE libshit)

target_include_directories(mse_tests PRIVATE include
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(mse_tests PRIVATE libshit)

target_include_directories(basic_tests PRIVATE include
    $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(basic_tests PRIVATE libshit)

add_custom_command(TARGET grad_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:libshit>
    $<TARGET_FILE_DIR:grad_tests>
)

add_custom_command(TARGET grad_loss_scaling_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:libshit>
    $<TARGET_FILE_DIR:grad_loss_scaling_tests>
)

add_custom_command(TARGET layers_model_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:libshit>
    $<TARGET_FILE_DIR:layers_model_tests>
)

add_custom_command(TARGET mse_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:libshit>
    $<TARGET_FILE_DIR:mse_tests>
)

add_custom_command(TARGET basic_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:libshit>
    $<TARGET_FILE_DIR:basic_tests>
)

add_test(NAME basic_tests COMMAND basic_tests)
add_test(NAME grad_tests COMMAND grad_tests)
add_test(NAME grad_loss_scaling_tests COMMAND grad_loss_scaling_tests)
add_test(NAME layers_model_tests COMMAND layers_model_tests)
add_test(NAME mse_tests COMMAND mse_tests)
