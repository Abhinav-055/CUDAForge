# CUDAForge

CUDAForge is a C++20 learning and benchmarking project with a portable CPU backend and an optional CUDA backend. It implements vector addition, reductions, dense matrix multiplication, and inclusive scan, with correctness tests and a benchmark harness that separates GPU kernel latency from data-transfer and end-to-end latency.

The project was built and tested on a **NVIDIA Tesla T4** using a CUDA-enabled Kaggle environment. The latest benchmark contains 28 measurements across five algorithm groups, with 101 repetitions per configuration.

**Repository:** [Abhinav-055/CUDAForge](https://github.com/Abhinav-055/CUDAForge)

## Layout

- `include/cudaforge/kernels.hpp` — backend interfaces
- `src/cpu/` — serial and OpenMP CPU implementations
- `src/cuda/` — CUDA kernels and baseline/optimized entry points
- `benchmarks/` — CPU and CUDA benchmark harnesses
- `tests/` — CPU and CUDA correctness tests
- `scripts/` — build and remote-environment helpers
- `docs/` — engineering report and technical notes
- `results/cudaforge_gpu_results_v3.csv` — measured Tesla T4 benchmark results

## Algorithms

| Operation | Work | Implementation |
|---|---:|---|
| Vector addition | O(n) | CPU implementation; CUDA grid-stride kernel with coalesced accesses |
| Sum and maximum | O(n) | CPU reductions; CUDA block reductions using shared memory |
| Dense matrix multiplication | O(mnk) | CPU i-k-j loop; CUDA naive and tiled shared-memory kernels |
| Inclusive scan | O(n) for the CPU implementation | CUDA quadratic baseline and tiled scan with block-offset accumulation |

### Implementation limitations

- The CUDA vector-add `optimized` entry point currently calls the same kernel as `naive`.
- The public CUDA sum and maximum baseline/optimized entry points share the same first-stage kernel implementation.
- The tiled scan's offset stage accumulates preceding tile totals for each output block. This is a correctness-oriented implementation, not a scalable hierarchical scan for very large inputs.
- These limitations are documented rather than presented as completed optimizations.

## Build and test

### CPU-only build

Requires CMake 3.20+ and a modern C++20 compiler.

```bash
cmake -S . -B build \
  -DENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On Windows, `scripts/build_cpu.ps1` provides the project's CPU build workflow.

OpenMP is optional. If unavailable, the CPU interfaces use their serial fallback.

### CUDA build

A CUDA-capable NVIDIA GPU, compatible driver, CUDA Toolkit including `nvcc`, CMake, and a compatible C++20 host compiler are required.

```bash
nvidia-smi
nvcc --version

cmake -S . -B build-cuda \
  -DENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda --parallel 2
ctest --test-dir build-cuda --output-on-failure
```

The CUDA build and both CPU and CUDA correctness tests were successfully run in the project's Kaggle Tesla T4 environment.

To reproduce the benchmark from the repository root:

```bash
./build-cuda/cudaforge_cuda_bench \
  101 results/cudaforge_gpu_results_v3.csv
```

The exact executable path may depend on the build directory and environment.

## Benchmark methodology

The CUDA benchmark uses warm-up iterations followed by repeated measurements and reports medians for:

- CUDA-event kernel latency
- Host-to-device (H2D) transfer latency
- Device-to-host (D2H) transfer latency
- End-to-end execution latency
- CPU baseline latency
- End-to-end speedup

End-to-end timing includes the relevant transfers and execution path, so it is not equivalent to kernel-only timing. The results are workload- and environment-dependent.

The latest CSV records the GPU model, input size, and repetition count. It contains 28 rows covering vector addition, sum, maximum, inclusive scan, and matrix multiplication.

See [`results/cudaforge_gpu_results_v3.csv`](results/cudaforge_gpu_results_v3.csv) and [`docs/engineering-report.md`](docs/engineering-report.md).

## Selected Tesla T4 results

Representative results from the 101-repetition benchmark:

| Operation | Input size | GPU kernel median (ms) | GPU end-to-end median (ms) | CPU median (ms) | End-to-end speedup |
|---|---:|---:|---:|---:|---:|
| Vector addition | 262,144 | 0.00957 | 0.91185 | 0.05670 | 0.062× |
| Sum | 262,144 | 0.02867 | 0.32747 | 0.08474 | 0.259× |
| Maximum | 262,144 | 0.01629 | 0.31028 | 0.08459 | 0.273× |
| Inclusive scan (tiled) | 262,144 | 0.25338 | 0.85137 | 0.17733 | 0.208× |
| Matrix multiplication (tiled) | 128 × 128 | 0.01683 | 0.09500 | 0.74083 | 7.798× |

Speedup is calculated as CPU median divided by GPU end-to-end median. Values below 1× mean the GPU path was slower for that workload; values above 1× mean the GPU path was faster.

These results illustrate why kernel latency alone is insufficient to assess GPU performance. Transfer overhead can outweigh the compute benefit for simple operations, while matrix multiplication can benefit substantially from parallel execution and shared-memory tiling.

The results are measurements from one Tesla T4 environment, not universal performance claims.

## Profiling and future work

On a CUDA host with NVIDIA profiling tools installed:

- Use Nsight Systems to investigate launch overhead, data transfers, synchronization, and timeline gaps.
- Use Nsight Compute to examine memory transactions, achieved bandwidth, occupancy, shared-memory utilization, and arithmetic throughput.

Priority improvements:

1. Implement genuinely distinct optimized vector-add and reduction kernels, then benchmark them.
2. Replace the scan offset accumulation with a hierarchical or recursive scan of block totals.
3. Profile representative workloads with Nsight Systems and Nsight Compute.
4. Repeat measurements on additional GPU architectures and larger inputs.
5. Investigate transfer strategies and workloads that keep data resident on the GPU.

CUDAForge is a measurement-oriented learning project. Performance claims should be tied to recorded results, and optimization labels should not be interpreted as proof of a performance improvement.