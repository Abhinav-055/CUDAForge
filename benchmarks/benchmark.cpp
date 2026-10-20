#include "cudaforge/kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#ifdef CUDAFORGE_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef _MSC_VER
#define CUDAFORGE_COMPILER_VERSION "MSVC " _CRT_STRINGIZE(_MSC_VER)
#elif defined(__VERSION__)
#define CUDAFORGE_COMPILER_VERSION __VERSION__
#else
#define CUDAFORGE_COMPILER_VERSION "unknown"
#endif

using clock_type=std::chrono::steady_clock;
template<class F> std::vector<double> measure(F&& f,int reps){std::vector<double> v;f();for(int r=0;r<reps;++r){auto a=clock_type::now();f();auto b=clock_type::now();v.push_back(std::chrono::duration<double,std::milli>(b-a).count());}std::sort(v.begin(),v.end());return v;}
static double median(const std::vector<double>&v){return v[v.size()/2];}
static double deviation(const std::vector<double>&v){double m=median(v),s=0;for(double x:v)s+=(x-m)*(x-m);return std::sqrt(s/v.size());}
int main(int argc,char**argv){int reps=argc>1?std::max(1,std::atoi(argv[1])):9;std::string file=argc>2?argv[2]:"cudaforge_results.csv";std::ofstream out(file);if(!out){std::cerr<<"Cannot open CSV: "<<file<<'\n';return 1;}out<<"algorithm,implementation,n,median_ms,stddev_ms,speedup_vs_serial,repetitions,compiler,build_type,cpu\n";std::cout<<"CUDAForge CPU benchmark; repetitions="<<reps<<"; compiler="<<CUDAFORGE_COMPILER_VERSION<<"\n";
 for(std::size_t n:{1024u,16384u,262144u}){std::vector<float>a(n,1.f),b(n,2.f),c(n),d(n);auto emit=[&](const std::string& alg,const std::string& impl,const std::vector<double>&t,double base){const char* cpu=std::getenv("PROCESSOR_IDENTIFIER");out<<alg<<','<<impl<<','<<n<<','<<std::setprecision(9)<<median(t)<<','<<deviation(t)<<','<<(base/median(t))<<','<<reps<<','<<CUDAFORGE_COMPILER_VERSION<<",Release,\""<<(cpu?cpu:"unknown")<<"\"\n";};
 auto add=measure([&]{cudaforge::cpu::vector_add(a.data(),b.data(),c.data(),n);},reps);auto addp=measure([&]{cudaforge::cpu::vector_add_parallel(a.data(),b.data(),c.data(),n);},reps);emit("vector_add","serial",add,median(add));emit("vector_add","openmp",addp,median(add));
 auto sum=measure([&]{volatile auto x=cudaforge::cpu::reduce_sum(a.data(),n);(void)x;},reps);auto sump=measure([&]{volatile auto x=cudaforge::cpu::reduce_sum_parallel(a.data(),n);(void)x;},reps);emit("sum","serial",sum,median(sum));emit("sum","openmp",sump,median(sum));
 auto mx=measure([&]{volatile auto x=cudaforge::cpu::reduce_max(a.data(),n);(void)x;},reps);auto mxp=measure([&]{volatile auto x=cudaforge::cpu::reduce_max_parallel(a.data(),n);(void)x;},reps);emit("maximum","serial",mx,median(mx));emit("maximum","openmp",mxp,median(mx));
 auto scan=measure([&]{cudaforge::cpu::inclusive_scan(a.data(),c.data(),n);},reps);auto scanp=measure([&]{cudaforge::cpu::inclusive_scan_parallel(a.data(),c.data(),n);},reps);emit("scan","serial",scan,median(scan));emit("scan","openmp",scanp,median(scan));
 auto side=std::min<std::size_t>(128,static_cast<std::size_t>(std::sqrt(static_cast<double>(n))));std::vector<float> ma(side*side,1),mb(side*side,2),mc(side*side);auto mm=measure([&]{cudaforge::cpu::matmul(ma.data(),mb.data(),mc.data(),side,side,side);},reps);auto mmp=measure([&]{cudaforge::cpu::matmul_parallel(ma.data(),mb.data(),mc.data(),side,side,side);},reps);emit("matmul","serial",mm,median(mm));emit("matmul","openmp",mmp,median(mm));
 }
#ifdef CUDAFORGE_ENABLE_CUDA
 int count=0;cudaError_t status=cudaGetDeviceCount(&count);if(status==cudaSuccess&&count>0){cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,0);std::cout<<"GPU="<<prop.name<<"\n";}else std::cout<<"CUDA build enabled but no usable CUDA device detected. GPU timings omitted.\n";
#endif
 std::cout<<"CPU CSV written to "<<file<<". Use cudaforge_cuda_bench for CUDA event timings.\n";return 0;}
