# libshit

A small CUDA-enabled C++ library with tensor operations, autograd-style gradient tape, and a lightweight layers API.

## Requirements

- Windows 10/11
- Visual Studio 2022 or newer with Desktop development for C++
- NVIDIA CUDA Toolkit 11.x or newer (tested with 13.2)
- CMake 3.18 or newer
- Ninja (recommended) or another supported CMake generator

## Build Instructions

1. Open a Visual Studio Developer PowerShell or x64 Native Tools command prompt.

2. Change into the repository root:

```powershell
cd "libshit"
```

3. Create the build directory if needed and configure the project:

```powershell
cmake -S . -B build -G Ninja
```

This will locate the CUDA toolkit via `find_package(CUDAToolkit REQUIRED)` and generate the build files.

4. Build the library and tests:

```powershell
cmake --build build --config Debug --target all
```

Alternatively, build a specific test target:

```powershell
cmake --build build --config Debug --target grad_tests -- -j1
```

## Run Tests

Run all CTest tests from the build directory:

```powershell
cd build
ctest -C Debug --output-on-failure
```

Or run a single test executable directly:

```powershell
.\layers_model_tests.exe
```

## Workspace Layout

- `include/`: public headers
- `src/`: library and CUDA kernel sources
- `tests/`: test executables
- `build/`: out-of-tree CMake build directory

## Notes

- The library target `libshit` is built as a shared DLL.
- CUDA headers are exposed to consumers so headers like `shit_errors.hpp` can compile in test targets.
- The project includes several tests:
  - `basic_tests`
  - `grad_tests`
  - `grad_loss_scaling_tests`
  - `layers_model_tests`
  - `mse_tests`
