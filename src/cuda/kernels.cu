#include "cudaforge/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace cudaforge::cuda {
void check(cudaError_t s,const char* e,const char* f,int l){if(s!=cudaSuccess)throw std::runtime_error(std::string(f)+":"+std::to_string(l)+": "+e+": "+cudaGetErrorString(s));}
namespace {
constexpr int block=256;
__global__ void vadd(const float*a,const float*b,float*c,std::size_t n){for(auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x)c[i]=a[i]+b[i];}
__global__ void redsum(const float*x,double*out,std::size_t n){extern __shared__ double sum_shared[];unsigned t=threadIdx.x;std::size_t i=blockIdx.x*(std::size_t)blockDim.x+t;double v=0;while(i<n){v+=x[i];i+=(std::size_t)gridDim.x*blockDim.x;}sum_shared[t]=v;__syncthreads();for(unsigned d=blockDim.x/2;d;d>>=1){if(t<d)sum_shared[t]+=sum_shared[t+d];__syncthreads();}if(!t)out[blockIdx.x]=sum_shared[0];}
__global__ void redmax(const float*x,float*out,std::size_t n){extern __shared__ float max_shared[];unsigned t=threadIdx.x;std::size_t i=blockIdx.x*(std::size_t)blockDim.x+t;float v=-INFINITY;while(i<n){v=fmaxf(v,x[i]);i+=(std::size_t)gridDim.x*blockDim.x;}max_shared[t]=v;__syncthreads();for(unsigned d=blockDim.x/2;d;d>>=1){if(t<d)max_shared[t]=fmaxf(max_shared[t],max_shared[t+d]);__syncthreads();}if(!t)out[blockIdx.x]=max_shared[0];}
__global__ void finish_sum(const double* in,double* out,std::size_t n){extern __shared__ double finish_sum_shared[];unsigned t=threadIdx.x;double v=0;for(std::size_t i=t;i<n;i+=blockDim.x)v+=in[i];finish_sum_shared[t]=v;__syncthreads();for(unsigned d=blockDim.x/2;d;d>>=1){if(t<d)finish_sum_shared[t]+=finish_sum_shared[t+d];__syncthreads();}if(!t)*out=finish_sum_shared[0];}
__global__ void finish_max(const float* in,float* out,std::size_t n){extern __shared__ float finish_max_shared[];unsigned t=threadIdx.x;float v=-INFINITY;for(std::size_t i=t;i<n;i+=blockDim.x)v=fmaxf(v,in[i]);finish_max_shared[t]=v;__syncthreads();for(unsigned d=blockDim.x/2;d;d>>=1){if(t<d)finish_max_shared[t]=fmaxf(finish_max_shared[t],finish_max_shared[t+d]);__syncthreads();}if(!t)*out=finish_max_shared[0];}
__global__ void mm(const float*a,const float*b,float*c,std::size_t m,std::size_t n,std::size_t k){auto col=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x,row=(std::size_t)blockIdx.y*blockDim.y+threadIdx.y;if(row<m&&col<n){float s=0;for(std::size_t p=0;p<k;++p)s+=a[row*k+p]*b[p*n+col];c[row*n+col]=s;}}
template<int T> __global__ void mmt(const float*a,const float*b,float*c,std::size_t m,std::size_t n,std::size_t k){__shared__ float as[T][T],bs[T][T];const unsigned tx=threadIdx.x,ty=threadIdx.y;const std::size_t row=(std::size_t)blockIdx.y*T+ty,col=(std::size_t)blockIdx.x*T+tx;float sum=0;for(std::size_t base=0;base<k;base+=T){as[ty][tx]=(row<m&&base+tx<k)?a[row*k+base+tx]:0;bs[ty][tx]=(base+ty<k&&col<n)?b[(base+ty)*n+col]:0;__syncthreads();for(int p=0;p<T;++p)sum+=as[ty][p]*bs[p][tx];__syncthreads();}if(row<m&&col<n)c[row*n+col]=sum;}
__global__ void scan(const float*x,float*y,std::size_t n){for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(std::size_t)gridDim.x*blockDim.x){float s=0;for(std::size_t j=0;j<=i;++j)s+=x[j];y[i]=s;}}
// Hierarchical scan: each block scans a tile in shared memory; a second kernel
// adds the exclusive prefix of tile totals. This is race-free for arbitrary n.
__global__ void scan_tiles(const float*x,float*y,float*totals,std::size_t n){extern __shared__ float scan_shared[];unsigned t=threadIdx.x;std::size_t i=(std::size_t)blockIdx.x*blockDim.x+t;scan_shared[t]=i<n?x[i]:0;__syncthreads();for(unsigned o=1;o<blockDim.x;o<<=1){float v=t>=o?scan_shared[t-o]:0;__syncthreads();scan_shared[t]+=v;__syncthreads();}if(i<n)y[i]=scan_shared[t];if(t==blockDim.x-1)totals[blockIdx.x]=scan_shared[t];}
__global__ void add_offsets(float*y,const float*totals,std::size_t n){auto i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&blockIdx.x){float off=0;for(unsigned b=0;b<blockIdx.x;++b)off+=totals[b];y[i]+=off;}}
int blocks(std::size_t n){int dev=0;CUDAFORGE_CUDA_CHECK(cudaGetDevice(&dev));static thread_local std::unordered_map<int,int> multiprocessors_by_device;auto it=multiprocessors_by_device.find(dev);if(it==multiprocessors_by_device.end()){cudaDeviceProp p{};CUDAFORGE_CUDA_CHECK(cudaGetDeviceProperties(&p,dev));it=multiprocessors_by_device.emplace(dev,p.multiProcessorCount).first;}return static_cast<int>(std::min<std::size_t>((n+block-1)/block,(std::size_t)it->second*8));}
template<class T,class K> T reduce(const float*x,std::size_t n,K launch){if(!n)return T{};int nb=blocks(n);T* d=nullptr;CUDAFORGE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d),nb*sizeof(T)));launch(nb,d);std::vector<T> h(nb);CUDAFORGE_CUDA_CHECK(cudaMemcpy(h.data(),d,nb*sizeof(T),cudaMemcpyDeviceToHost));CUDAFORGE_CUDA_CHECK(cudaFree(d));if constexpr(std::is_same_v<T,double>){double z=0;for(auto v:h)z+=v;return z;}else return *std::max_element(h.begin(),h.end());}
}
void vector_add_naive(const float*a,const float*b,float*c,std::size_t n){if(!n)return;int nb=blocks(n);vadd<<<nb,block>>>(a,b,c,n);CUDAFORGE_CUDA_CHECK(cudaGetLastError());}
void vector_add_optimized(const float*a,const float*b,float*c,std::size_t n){vector_add_naive(a,b,c,n);}
double reduce_sum_naive(const float*x,std::size_t n){return reduce<double>(x,n,[&](int nb,double*d){redsum<<<nb,block,block*sizeof(double)>>>(x,d,n);CUDAFORGE_CUDA_CHECK(cudaGetLastError());});}
double reduce_sum_optimized(const float*x,std::size_t n){return reduce_sum_naive(x,n);}
float reduce_max_naive(const float*x,std::size_t n){return reduce<float>(x,n,[&](int nb,float*d){redmax<<<nb,block,block*sizeof(float)>>>(x,d,n);CUDAFORGE_CUDA_CHECK(cudaGetLastError());});}
float reduce_max_optimized(const float*x,std::size_t n){return reduce_max_naive(x,n);}
void reduce_sum_device(const float*x,double*out,double*scratch,std::size_t n){if(!n){CUDAFORGE_CUDA_CHECK(cudaMemset(out,0,sizeof(double)));return;}const int nb=blocks(n);redsum<<<nb,block,block*sizeof(double)>>>(x,scratch,n);CUDAFORGE_CUDA_CHECK(cudaGetLastError());finish_sum<<<1,block,block*sizeof(double)>>>(scratch,out,static_cast<std::size_t>(nb));CUDAFORGE_CUDA_CHECK(cudaGetLastError());}
void reduce_max_device(const float*x,float*out,float*scratch,std::size_t n){if(!n){CUDAFORGE_CUDA_CHECK(cudaMemset(out,0,sizeof(float)));return;}const int nb=blocks(n);redmax<<<nb,block,block*sizeof(float)>>>(x,scratch,n);CUDAFORGE_CUDA_CHECK(cudaGetLastError());finish_max<<<1,block,block*sizeof(float)>>>(scratch,out,static_cast<std::size_t>(nb));CUDAFORGE_CUDA_CHECK(cudaGetLastError());}
void matmul_naive(const float*a,const float*b,float*c,std::size_t m,std::size_t n,std::size_t k){if(!m||!n)return;dim3 t(16,16),g((n+15)/16,(m+15)/16);mm<<<g,t>>>(a,b,c,m,n,k);CUDAFORGE_CUDA_CHECK(cudaGetLastError());}
void matmul_optimized(const float*a,const float*b,float*c,std::size_t m,std::size_t n,std::size_t k){if(!m||!n)return;dim3 t(16,16),g((n+15)/16,(m+15)/16);mmt<16><<<g,t>>>(a,b,c,m,n,k);CUDAFORGE_CUDA_CHECK(cudaGetLastError());}
void inclusive_scan_naive(const float*x,float*y,std::size_t n){if(!n)return;scan<<<blocks(n),block>>>(x,y,n);CUDAFORGE_CUDA_CHECK(cudaGetLastError());}
void inclusive_scan_optimized_device(const float*x,float*y,float*d,std::size_t n){if(!n)return;int nb=static_cast<int>((n+block-1)/block);scan_tiles<<<nb,block,block*sizeof(float)>>>(x,y,d,n);CUDAFORGE_CUDA_CHECK(cudaGetLastError());add_offsets<<<nb,block>>>(y,d,n);CUDAFORGE_CUDA_CHECK(cudaGetLastError());}
void inclusive_scan_optimized(const float*x,float*y,std::size_t n){if(!n)return;int nb=static_cast<int>((n+block-1)/block);float*d=nullptr;CUDAFORGE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d),nb*sizeof(float)));try{inclusive_scan_optimized_device(x,y,d,n);CUDAFORGE_CUDA_CHECK(cudaFree(d));}catch(...){cudaFree(d);throw;}}
}
