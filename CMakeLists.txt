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
    src/save/save.cpp
    include/save/save.hpp
    include/libshit.h
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
file(GLOB TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})

    target_include_directories(${test_name} PRIVATE include
        $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(${test_name} PRIVATE libshit)

    add_custom_command(TARGET ${test_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:libshit>
        $<TARGET_FILE_DIR:${test_name}>
    )

    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
