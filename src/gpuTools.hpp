#ifndef GPU_TOOLS_HPP
#define GPU_TOOLS_HPP



#include <sstream>
#include <cuda_runtime.h>

#include "Logger.hpp"
#include "cudaPrefix.hpp"



namespace microflow
{



void initializeGPU(int gpuId ) ;
void finalize() ;

void cudaCheck( cudaError_t cudaCode, std::string file, size_t line ) ;

#define CUDA_CHECK(code)   cudaCheck( (code), __FILE__, __LINE__ ) ;



}



#endif
