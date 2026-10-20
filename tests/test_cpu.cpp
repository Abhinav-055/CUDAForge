#include "cudaforge/kernels.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(){try{
 std::vector<float>a{1,2,3,4,5},b{5,4,3,2,1},o(5);cudaforge::cpu::vector_add(a.data(),b.data(),o.data(),5);require(o==std::vector<float>(5,6),"vector add");cudaforge::cpu::vector_add_parallel(a.data(),b.data(),o.data(),5);require(o[4]==6,"parallel vector add");
 require(cudaforge::cpu::reduce_sum(a.data(),5)==15,"sum");require(cudaforge::cpu::reduce_sum_parallel(a.data(),5)==15,"parallel sum");require(cudaforge::cpu::reduce_max_parallel(a.data(),5)==5,"max");
 float ma[]{1,2,3,4,5,6},mb[]{7,8,9,10,11,12},mc[4]{};cudaforge::cpu::matmul(ma,mb,mc,2,2,3);require(mc[0]==58&&mc[1]==64&&mc[2]==139&&mc[3]==154,"matmul");
 float scan[5]{},scan2[5]{};cudaforge::cpu::inclusive_scan(a.data(),scan,5);cudaforge::cpu::inclusive_scan_parallel(a.data(),scan2,5);require(scan[4]==15&&scan2[4]==15&&scan2[2]==6,"scan");cudaforge::cpu::inclusive_scan(a.data(),scan,0);std::cout<<"All CPU correctness checks passed.\n";
 }catch(const std::exception&e){std::cerr<<"Test failure: "<<e.what()<<'\n';return 1;}return 0;}
