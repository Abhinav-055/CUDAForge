#pragma once

#include <cstddef>
#include <vector>

namespace cudaforge::cpu {
void vector_add(const float* a, const float* b, float* out, std::size_t n);
void vector_add_parallel(const float* a, const float* b, float* out, std::size_t n);
double reduce_sum(const float* data, std::size_t n);
double reduce_sum_parallel(const float* data, std::size_t n);
float reduce_max(const float* data, std::size_t n);
float reduce_max_parallel(const float* data, std::size_t n);
void matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n, std::size_t k);
void matmul_parallel(const float* a, const float* b, float* c, std::size_t m, std::size_t n, std::size_t k);
void inclusive_scan(const float* input, float* output, std::size_t n);
void inclusive_scan_parallel(const float* input, float* output, std::size_t n);
}

#ifdef CUDAFORGE_ENABLE_CUDA
#include <cuda_runtime_api.h>
namespace cudaforge::cuda {
void check(cudaError_t status, const char* expression, const char* file, int line);
#define CUDAFORGE_CUDA_CHECK(expr) ::cudaforge::cuda::check((expr), #expr, __FILE__, __LINE__)
void vector_add_naive(const float*, const float*, float*, std::size_t);
void vector_add_optimized(const float*, const float*, float*, std::size_t);
double reduce_sum_naive(const float*, std::size_t);
double reduce_sum_optimized(const float*, std::size_t);
float reduce_max_naive(const float*, std::size_t);
float reduce_max_optimized(const float*, std::size_t);
// Caller supplies device output scalars and scratch arrays of at least n values.
// These launch-only variants enable CUDA-event timing without allocation or copies.
void reduce_sum_device(const float*, double* output, double* scratch, std::size_t n);
void reduce_max_device(const float*, float* output, float* scratch, std::size_t n);
void matmul_naive(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
void matmul_optimized(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
void inclusive_scan_naive(const float*, float*, std::size_t);
void inclusive_scan_optimized(const float*, float*, std::size_t);
void inclusive_scan_optimized_device(const float*, float*, float* scratch, std::size_t);
}
#endif
