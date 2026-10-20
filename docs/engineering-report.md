# CUDAForge engineering report

## Status

The project separates CPU and CUDA interfaces and supports a CPU-only CMake configuration. CPU algorithms include serial and OpenMP-capable variants for vector addition, sum, maximum, matrix multiplication, and inclusive scan. CUDA source implements actual kernels for all four requested algorithm groups, with boundary checks, shared-memory block reductions, tiled matrix multiplication, and a tiled scan. A CUDA-versus-CPU correctness executable is provided. CUDA error checks cover launch and runtime API errors.

### CUDA audit

| Feature | Implemented | Incomplete or unverified |
|---|---|---|
| Vector addition | Real grid-stride CUDA kernel; non-multiple sizes handled | `optimized` currently calls the same kernel as `naive`; GPU test/benchmark not run |
| Sum / maximum | Shared-memory block reduction; launch-only two-stage device entry points for timing | Public `optimized` and `naive` wrappers currently select the same first-stage kernel; GPU test/benchmark not run |
| Matrix multiplication | Naive output-per-thread and shared-memory tiled kernel; bounds checks cover tails | No GPU run yet; performance/occupancy trade-offs unmeasured |
| Inclusive scan | Quadratic naive baseline and shared-memory tiled scan with block offsets | Offset kernel sums all earlier block totals per output (quadratic in block count); no GPU run yet |
| Error checks | CUDA API and launch checks are present; no-device CUDA test returns CTest skip (77) | Exception cleanup and runtime failure paths have not been exercised on a GPU |
| GPU benchmark | CUDA events time kernel launches; wall clock times paired synchronous H2D, D2H, and end-to-end paths; default 101 samples with warm-up; CSV includes GPU, sizes, reps, CPU baseline, and speedup | Built and executed only on a CUDA machine; never run locally because no `nvcc`/GPU |

The CUDA source is actual CUDA C++, not a CPU simulation. Presence of source and the CMake target does not establish that it compiles or works on hardware.

## Measurements

The checked-in `cudaforge_results.csv` contains an actual local Release CPU run (3 repetitions, MSVC 19.44, Intel64 Family 6 Model 170). The CPU CMake build and CTest pass on this machine. Timings are short local observations with only three samples; they are not broadly representative. OpenMP overhead loses on small sizes while larger array operations and matrix multiplication show benefits in this run. The CUDA configure check fails as expected because `nvcc` is absent. A separate CUDA benchmark now records CUDA-event kernel/H2D/D2H medians and wall-clock end-to-end time, but neither that executable nor the CUDA correctness test could be compiled or run here. GPU timings and GPU correctness remain unverified.

## Trade-offs and next validation

Tiled matrix multiplication reuses operand tiles in shared memory, reducing redundant global reads at the cost of synchronization and shared-memory occupancy. Reductions coalesce input accesses and combine values within blocks. Vector addition is bandwidth-bound with little arithmetic per byte, so transfer and launch costs dominate small sizes. The simple CUDA scan is intentionally a slow baseline; the tiled scan is faster in work complexity within a tile, but its per-block offset accumulation scales with tile count. A recursive scan of block totals is needed for large workloads. CUDA “optimized” reduction and vector-add entry points currently call the same kernel as their baseline and should not be presented as optimization wins.

Next: on the target L4, run the Camber script, verify CTest reports the CUDA test as passed (not skipped), inspect the emitted GPU CSV, then profile representative cases with Nsight Systems/Compute. Record actual environment details and raw CSV before writing performance conclusions.
