#! /bin/bash

# script stops as soon as something fails
step -e


cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure