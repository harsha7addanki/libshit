# libshit - Copilot Instructions

## Build & Test

```powershell
# Configure with Ninja
cmake -S . -B build -G Ninja

# Build everything (Debug)
cmake --build build --config Debug --target all

# Build a specific test
cmake --build build --config Debug --target grad_tests -- -j1

# Run all tests
cd build
ctest -C Debug --output-on-failure

# Run a single test executable directly
.\build\basic_tests.exe
```

**Test targets:** `basic_tests`, `grad_tests`, `grad_loss_scaling_tests`, `layers_model_tests`, `mse_tests`, `save_load_test`

**Requirements:** Visual Studio 2022+, NVIDIA CUDA Toolkit 11.x+, Ninja. Open a Visual Studio Developer PowerShell before building.

## Architecture

Three-tier design in the `libshit` namespace:

### 1. Core (`libshit::core`)
- **`ShitTensor`** — Row-major `float` tensor with separate CPU (`cudaHostAlloc`) and GPU (`cudaMalloc`) pointers. Uses `to_gpu()`/`to_cpu()` for explicit transfers (no unified memory). Supports `{3, 4}` initializer list or `vector<int64_t>` shapes. Copy is deleted — pass by pointer.
- **`ShitTape`** — Singleton gradient tape. `start()` clears the node list, `stop()` ends recording, `backward(registry)` replays recorded nodes in reverse. Each op constructor checks `ShitTape::instance().is_active()` and pushes a node if the tape is recording.
- **`ShitGradientRegistry`** — Manages gradient tensors keyed by operand pointer. `get_grad(tensor)` lazily allocates a zero-initialized gradient. Owns and frees gradient memory in destructor.
- **Forward ops** (e.g. `relu`, `matmul`, `add`, `mse`) allocate output tensors, optionally record tape nodes, then launch CUDA kernels. Defined in `src/shit_kernels.cu`.
- **Backward kernels** (e.g. `backward_relu`, `backward_matmul`) are in `src/shit_gradients.cu`. All use `atomicAdd` to accumulate gradients since multiple paths may contribute.

### 2. Layers (`libshit::layers`)
- **`ShitLayer`** — Base class with `register_parameter(name, shape, init)` for weight creation, `add<T>(args...)` for composing child layers, `build(input)` / `call(input)` pattern (lazy-initialization on first forward pass), and recursive `get_parameters()`.
- **`ShitModel`** — Extends `ShitLayer` with training loop: `train_step` starts the tape, runs forward, sets loss gradient to 1.0, calls `backward(registry)`, then steps optimizer per parameter. `train(data, epochs)` iterates dataset with batch logging.
- **Built-in layers:** `Dense(in, out)` with Xavier-uniform init, `ReLU` (no parameters).

### 3. Optimizer (`libshit::optim`)
- **`ShitOptimizer`** — Abstract base with `step(param, grad)`. **`SGD`** implements SGD weight update. CUDA kernel in `src/optim/optimizer.cu`.

### 4. Serialization (`libshit::save`)
- **`ShitSerializer::save(model, path)`** — Writes binary `"SHIT"` magic, version uint32, then raw float data per parameter.
- **`ShitSerializer::load(model, path)`** — Reads magic and version, then reads parameter data directly into pinned host memory via `cpu_ptr()` and transfers to GPU.

## Key Conventions

- **Explicit memory management:** All `ShitTensor*` are raw pointers allocated with `new` and freed with `free_tensor()`. Tensors are non-copyable. Always manually transfer data with `to_gpu()` / `to_cpu()`.
- **Tape node pattern:** Each operator requires: (1) a `__global__` forward kernel, (2) a C++ function wrapper that checks the tape and launches the kernel, (3) a `ShitOperatorNode` subclass with `forward()`/`backward()` for autodiff, (4) a `__global__` backward kernel that uses `atomicAdd` for gradient accumulation. All four must be added in sync across `shit_tensor.hpp`, `shit_gradients.hpp`, `shit_kernels.cu`, `shit_gradients.cu`, and `shit_gradients.cpp`.
- **`SHIT_API`** — All public functions/classes exported from the DLL need this macro. Internal/static functions do not.
- **`SHIT_CHECK(ans)`** macro wraps every CUDA runtime call for error checking.
- **Thread/block config:** 1D kernels use 256 threads/block. 2D kernels use 16x16 threads. Block count is always `(size + threads - 1) / threads`.
- **Tests** are standalone `.cpp` files in `tests/`. Each is a full executable with `main()`. They return 0 on pass, 1 on fail. The `CMakeLists.txt` globs all `tests/*.cpp` into separate test targets automatically.
- **CUDA in headers:** Unlike typical CUDA projects, CUDA headers (e.g. `cuda_runtime.h`) are included in public headers, so CUDA toolkit must be available to all consumers.
- **No namespaces by default:** The `LIBSHIT_USE_NAMESPACE` and `LIBSHIT_CORE_USE_NAMESPACE` macros are opt-in.
- **File extension matters:** `.cu` files contain CUDA kernels; `.cpp` files contain host-only logic. The `CMakeLists.txt` lists each source file explicitly (no globbing for sources).

## VS Code Setup

Configured with CMake Tools extension, Ninja generator, and CUDA 13.2 toolkit path. See `.vscode/settings.json` and `c_cpp_properties.json`.