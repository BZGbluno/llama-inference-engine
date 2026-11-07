# llama-inference-engine

An LLM inference engine in C++/CUDA.

## Layout

```
├── CMakeLists.txt
├── include/   tensor/ model/ runtime/ cuda/ server/
├── src/       tensor/ model/ runtime/ cuda/ server/
├── tests/
├── benchmarks/
├── examples/
└── scripts/
```

Any `.cpp`/`.cu` file added under `src/` is compiled into the `llm_engine` library, and any
`.cpp` under `tests/` is built into the GoogleTest binary. Re-running CMake is not required.

## Dependencies

| Dependency  | Source |
|-------------|--------|
| C++17 compiler | system |
| CMake ≥ 3.24 | system |
| CUDA Toolkit | system; **12.8+ for RTX 50-series (sm_120)** |
| cuBLAS | ships with the CUDA Toolkit |
| GoogleTest 1.15.2 | fetched by CMake |
| safetensors | parsed via nlohmann/json 3.11.3 (fetched by CMake) |

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```
