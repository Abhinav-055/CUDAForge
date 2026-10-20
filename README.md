# CUDAForge

CUDAForge is a C++20 learning and measurement project with a portable CPU backend and an optional, real CUDA backend. CPU-only configuration does not search for or include CUDA headers. CUDA functions accept **device pointers**; callers are responsible for transfers and allocation. The checked-in CSV contains measured CPU timings from the local Release build, not illustrative values.

## Layout

- `include/cudaforge/kernels.hpp`: backend interfaces
- `src/cpu/`: serial and OpenMP CPU algorithms
- `src/cuda/`: CUDA kernels (naive and optimized entry points)
- `benchmarks/`: CSV CPU benchmark harness
- `tests/`: CPU correctness checks
- `scripts/`: local/remote build helpers
- `docs/`: algorithm, profiling, and results notes

## Algorithms

| Operation | Work | Implementation notes |
|---|---:|---|
| Vector add | O(n) | Coalesced CUDA loads/stores; grid-stride loop |
| Sum / maximum | O(n) | CPU reductions; CUDA block reductions use shared memory |
| Dense matrix multiply | O(mnk) | CPU i-k-j loop; CUDA naive output-per-thread and tiled shared-memory kernel |
| Inclusive scan | O(n) | CPU chunked scan; CUDA simple O(n²) baseline and shared-memory tiled scan |

The CUDA scan's tile-offset pass currently scans prior tile totals per output block. It is a correctness-oriented hierarchy, not a scalable multi-level scan for very large inputs. Likewise the CUDA “optimized” vector add and reductions currently share their baseline kernel; these names preserve comparison slots until a separately measured improvement is implemented. Verify every proposed optimization before drawing performance conclusions.

## Build and test

Windows CPU-only (PowerShell):

```powershell
./scripts/build_cpu.ps1
```

Equivalent commands on Windows, Linux, or macOS with a C++20 compiler and CMake 3.20+:

```sh
cmake -S . -B build -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

OpenMP is optional. If CMake cannot find it, parallel-named CPU interfaces execute the serial fallback. A current C++20 toolchain is required. For example, use a recent MSVC, GCC, or Clang; MinGW GCC 6.3 is too old for this project's C++20 requirement.

## CUDA on a remote NVIDIA L4

Use a Linux Camber allocation with an attached NVIDIA L4, compatible NVIDIA driver, CUDA Toolkit (including `nvcc`), and a C++20 host compiler. From the repository root, run `bash scripts/camber_cuda.sh`; it checks the device/toolkit, configures `ENABLE_CUDA=ON`, builds, runs CTest, and exports the GPU benchmark CSV. It expects the Toolkit and CMake to be available on the allocated machine; it does not install them.

```sh
nvidia-smi
nvcc --version
bash scripts/camber_cuda.sh 9 cudaforge_gpu_results.csv
```

Or configure with `cmake -S . -B build-cuda -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release`, build, and run tests. The explicit CUDA option fails with a setup message when `nvcc` is unavailable. CUDA has not been compiled or run as part of the local CPU-only workflow; do not treat CUDA results as validated until this sequence succeeds on an actual GPU.

## Benchmarks

The CPU executable `cudaforge_bench` measures CPU implementations. In a CUDA build, `cudaforge_cuda_bench [repetitions] [output.csv]` defaults to 101 timed samples (after warm-up) and reports medians for CUDA-event kernel time, wall-clock H2D and D2H copies, and wall-clock end-to-end time, alongside a CPU parallel baseline and end-to-end speedup. Transfer timing includes the host-side staging cost of synchronous copies; each H2D value measures the complete input set together. CSV rows include GPU model, input size, and repetition count. Input `n` is vector length for vector/reduction/scan and matrix side length for matrix multiplication. The quadratic naive scan is only measured up to 4096 elements. GPU measurements have not been run on this development machine; the output file should be generated on the target GPU and retained with that run's environment details.

## Profiling

On a CUDA host, use Nsight Systems (`nsys profile --stats=true ./build-cuda/cudaforge_bench`) to inspect launch/transfer overhead and timeline gaps. Use Nsight Compute (`ncu --set full ./build-cuda/cudaforge_bench`) for achieved bandwidth, memory transactions/coalescing, occupancy, shared-memory use, and arithmetic throughput. If either tool is unavailable, record that fact and use event timing plus `nvidia-smi`; profiler absence does not prevent correctness testing.

## Engineering notes

The report at [`docs/engineering-report.md`](docs/engineering-report.md) records current status and interpretation limits. CPU timing is sensitive to frequency scaling, background load, OpenMP startup, and cache residency. GPU kernel-only time excludes transfers and launch latency; report both kernel and end-to-end time when adding CUDA measurements. No sample speedup numbers are supplied because no runs have been made.
