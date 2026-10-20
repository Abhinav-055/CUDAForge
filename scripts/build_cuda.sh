#!/usr/bin/env bash
set -euo pipefail
nvidia-smi
nvcc --version
cmake -S . -B build-cuda -DENABLE_CUDA=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
./build-cuda/cudaforge_cuda_bench 9 cudaforge_gpu_results.csv
