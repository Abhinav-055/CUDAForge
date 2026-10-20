# CUDAForge Engineering Report

## 1. Project status

CUDAForge is a C++20 project with separate CPU and CUDA interfaces. The CPU-only configuration can be built without CUDA headers or a CUDA Toolkit. The CUDA backend implements actual GPU kernels rather than simulating GPU operations on the CPU.

Implemented algorithm groups:

- Vector addition
- Sum reduction
- Maximum reduction
- Dense matrix multiplication
- Inclusive scan

The repository includes CMake build targets, correctness tests, CPU and CUDA benchmark harnesses, and CSV output.

### Validation environment

| Item | Recorded environment |
|---|---|
| GPU | NVIDIA Tesla T4 |
| CUDA Toolkit | 12.8.93 (`nvcc`) |
| Host compiler | GCC 13.3 |
| NVIDIA driver | 580.178.04 |
| Benchmark repetitions | 101 per configuration |
| CUDA correctness test | Passed |
| CPU correctness test | Passed |

The driver reports CUDA compatibility through version 13.0. The Toolkit used to compile the project was CUDA 12.8.93.

The successful CUDA build and test run establishes that the implementation compiled and passed the repository's current correctness tests in this environment. It does not establish correctness for every possible input size or prove that all kernels are optimally implemented.

## 2. CUDA implementation audit

| Feature | Implemented | Limitation or follow-up |
|---|---|---|
| Vector addition | Grid-stride CUDA kernel with boundary handling | `optimized` currently calls the same kernel as `naive` |
| Sum and maximum | Shared-memory block reduction and multi-stage device entry points | Public baseline and optimized wrappers share their first-stage kernel |
| Matrix multiplication | Naive output-per-thread kernel and shared-memory tiled kernel | Occupancy and performance trade-offs have not been profiled with Nsight |
| Inclusive scan | Quadratic naive baseline and tiled shared-memory scan | Offset stage accumulates earlier block totals for each output block |
| Error handling | CUDA runtime and launch checks; CUDA tests skip when no device is available | Failure-path and resource-cleanup behavior deserves further testing |
| Benchmarking | CUDA-event kernel timing and wall-clock transfer/end-to-end timing | Results currently cover one GPU environment |

### Algorithm considerations

**Vector addition:** Each output element requires two input reads and one output write. The operation performs little arithmetic relative to its memory traffic, so it is bandwidth-bound for sufficiently large inputs. Kernel-launch and transfer costs can dominate for small arrays.

**Reductions:** Each thread processes input values, and threads within a block combine partial results using shared memory. Larger inputs require partial results to be combined across blocks. A genuinely optimized reduction could improve data reuse, instruction efficiency, or synchronization behavior; the current `optimized` entry point should not be described as an independent performance improvement.

**Matrix multiplication:** The naive implementation assigns output work to threads, while the tiled implementation reuses input tiles in shared memory. Reuse reduces redundant global-memory accesses but introduces synchronization and shared-memory resource costs.

**Inclusive scan:** The naive CUDA baseline has quadratic work. The tiled implementation performs scan work within tiles, then applies block offsets. Its offset stage currently sums prior block totals for each output block, which limits scalability. A hierarchical scan over block totals is a clear future improvement.

## 3. Benchmark methodology

The latest artifact is `results/cudaforge_gpu_results_v3.csv`.

The benchmark uses warm-up iterations and 101 timed repetitions per configuration. It records medians for:

- CUDA-event kernel time
- H2D transfer time
- D2H transfer time
- End-to-end time
- CPU baseline time
- End-to-end speedup

The benchmark includes five algorithm groups and 28 rows. Vector addition, reductions, and scan use vector length as input size; matrix multiplication uses the matrix side length.

The end-to-end measurement includes data movement and execution. It must be distinguished from kernel-only latency, which does not represent the cost of moving data from host memory to the GPU and retrieving results.

## 4. Measured results

Selected values from the Tesla T4 v3 CSV:

| Operation | Size | Kernel (ms) | H2D (ms) | D2H (ms) | End-to-end (ms) | CPU (ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector add, naive | 262,144 | 0.00957 | 0.57901 | 0.31882 | 0.91185 | 0.05670 | 0.062× |
| Sum | 262,144 | 0.02867 | 0.28720 | 0.00807 | 0.32747 | 0.08474 | 0.259× |
| Maximum | 262,144 | 0.01629 | 0.28396 | 0.00798 | 0.31028 | 0.08459 | 0.273× |
| Inclusive scan, tiled | 262,144 | 0.25338 | 0.28518 | 0.30080 | 0.85137 | 0.17733 | 0.208× |
| Matrix multiplication, tiled | 128 × 128 | 0.01683 | 0.05532 | 0.02682 | 0.09500 | 0.74083 | 7.798× |

Speedup is defined as:

\[
\text{Speedup} =
\frac{\text{CPU median time}}{\text{GPU end-to-end median time}}
\]

A speedup below 1× means the end-to-end GPU path is slower than the CPU baseline. A value above 1× means the GPU path is faster for that measured case.

### Interpretation

- Vector addition, reductions, and scan do not outperform the CPU end-to-end in the selected 262,144-element cases.
- For vector addition, the measured transfer costs are much larger than kernel time.
- Tiled matrix multiplication at 128 × 128 is substantially faster end-to-end than its CPU baseline in this run.
- These results are specific to the recorded hardware, software stack, benchmark methodology, and input sizes.
- Kernel-only performance must not be presented as end-to-end speedup.

## 5. Timing anomaly and measurement discipline

An earlier benchmark run produced unexpectedly high kernel timings for simple CUDA operations. The subsequent benchmark revision reported substantially lower timings after the timing-path changes.

This discrepancy is a useful reminder that benchmark harness correctness is part of performance engineering. A timing result is meaningful only when the measurement boundaries, synchronization behavior, warm-up, and execution path are understood.

The v3 CSV should be treated as the latest recorded result set. Comparisons between revisions should be described as changes in measured kernel timing, not as equivalent changes in end-to-end speedup.

## 6. Limitations

- The `optimized` vector-add and reduction entry points currently share kernels with their baseline counterparts.
- The tiled scan's offset stage scales poorly as the number of tiles increases.
- Only one GPU model is represented in the latest recorded results.
- Nsight Systems and Nsight Compute profiling results have not yet been recorded.
- The tests validate the tested inputs and execution paths, not every possible boundary, overflow, resource-exhaustion, or runtime-failure condition.
- Timing results may vary with system load, clock behavior, cache residency, transfer staging, and compiler settings.

## 7. Recommended next work

1. Implement and benchmark distinct optimized vector-add and reduction kernels.
2. Replace the scan offset pass with a scalable hierarchical scan.
3. Profile representative kernels with Nsight Systems and Nsight Compute.
4. Expand boundary-case and randomized correctness testing.
5. Evaluate larger workloads and additional GPU architectures.
6. Preserve raw CSVs and record the exact build environment for every benchmark revision.

## Conclusion

CUDAForge demonstrates a working CPU/CUDA project, automated correctness checks, and a benchmark harness that distinguishes kernel latency from data-transfer and end-to-end cost. Its measured results show that GPU acceleration depends on the workload: transfer overhead can dominate simple operations, while tiled matrix multiplication can benefit substantially from GPU execution.

The project is best presented as an engineering and measurement project with clearly documented limitations and concrete future optimization work—not as a claim that every CUDA implementation is faster than its CPU counterpart.