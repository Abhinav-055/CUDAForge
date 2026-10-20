#include "cudaforge/kernels.hpp"
#include <algorithm>
#include <limits>
#ifdef CUDAFORGE_HAS_OPENMP
#include <omp.h>
#endif

namespace cudaforge::cpu {
void vector_add(const float* a,const float* b,float* o,std::size_t n) { for(std::size_t i=0;i<n;++i)o[i]=a[i]+b[i]; }
void vector_add_parallel(const float* a,const float* b,float* o,std::size_t n) {
#ifdef CUDAFORGE_HAS_OPENMP
#pragma omp parallel for
#endif
 for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(n);++i)o[i]=a[i]+b[i]; }
double reduce_sum(const float* x,std::size_t n) { double s=0;for(std::size_t i=0;i<n;++i)s+=x[i];return s; }
double reduce_sum_parallel(const float* x,std::size_t n) { double s=0;
#ifdef CUDAFORGE_HAS_OPENMP
#pragma omp parallel for reduction(+:s)
#endif
 for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(n);++i)s+=x[i];return s; }
float reduce_max(const float* x,std::size_t n) { float m=-std::numeric_limits<float>::infinity();for(std::size_t i=0;i<n;++i)m=std::max(m,x[i]);return m; }
float reduce_max_parallel(const float* x,std::size_t n) { float m=-std::numeric_limits<float>::infinity();
#ifdef CUDAFORGE_HAS_OPENMP
#pragma omp parallel
 { float local=-std::numeric_limits<float>::infinity();
#pragma omp for nowait
   for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(n);++i)local=std::max(local,x[i]);
#pragma omp critical
   {m=std::max(m,local);} }
#else
 for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(n);++i)m=std::max(m,x[i]);
#endif
 return m; }
void matmul(const float* a,const float* b,float* c,std::size_t m,std::size_t n,std::size_t k) {
 std::fill(c,c+m*n,0.f);for(std::size_t i=0;i<m;++i)for(std::size_t p=0;p<k;++p){const float v=a[i*k+p];for(std::size_t j=0;j<n;++j)c[i*n+j]+=v*b[p*n+j];} }
void matmul_parallel(const float* a,const float* b,float* c,std::size_t m,std::size_t n,std::size_t k) {
#ifdef CUDAFORGE_HAS_OPENMP
#pragma omp parallel for
#endif
 for(std::ptrdiff_t ii=0;ii<static_cast<std::ptrdiff_t>(m);++ii){const auto i=static_cast<std::size_t>(ii);for(std::size_t j=0;j<n;++j){float sum=0;for(std::size_t p=0;p<k;++p)sum+=a[i*k+p]*b[p*n+j];c[i*n+j]=sum;}} }
void inclusive_scan(const float* x,float* y,std::size_t n){float s=0;for(std::size_t i=0;i<n;++i)y[i]=s+=x[i];}
void inclusive_scan_parallel(const float* x,float* y,std::size_t n){
#ifdef CUDAFORGE_HAS_OPENMP
 const auto workers=static_cast<std::size_t>(omp_get_max_threads());
#else
 const std::size_t workers=1;
#endif
 if(workers<2||n<workers*1024){inclusive_scan(x,y,n);return;}const std::size_t chunk=(n+workers-1)/workers;std::vector<double> sums(workers,0.0);
#ifdef CUDAFORGE_HAS_OPENMP
#pragma omp parallel for
#endif
 for(std::ptrdiff_t wi=0;wi<static_cast<std::ptrdiff_t>(workers);++wi){const auto w=static_cast<std::size_t>(wi), begin=w*chunk,end=std::min(n,begin+chunk);double s=0;for(auto i=begin;i<end;++i)s+=x[i];sums[w]=s;}
 for(std::size_t w=1;w<workers;++w)sums[w]+=sums[w-1];
#ifdef CUDAFORGE_HAS_OPENMP
#pragma omp parallel for
#endif
 for(std::ptrdiff_t wi=0;wi<static_cast<std::ptrdiff_t>(workers);++wi){const auto w=static_cast<std::size_t>(wi),begin=w*chunk,end=std::min(n,begin+chunk);float s=w?static_cast<float>(sums[w-1]):0.f;for(auto i=begin;i<end;++i)y[i]=s+=x[i];}
}
}
