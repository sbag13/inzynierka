#include "cuda_profiler_api.h"

#include "gpuTools.hpp"
#include "Logger.hpp"
#include "Exceptions.hpp"



namespace microflow
{



void printDeviceProperties( cudaDeviceProp devProp )
{
	double clockRateInGhz = devProp.clockRate / 1.0E6 ;
	double globalMemoryInMB = devProp.totalGlobalMem / (1024.0 * 1024.0) ; //FIXME: check for overflow

	logger << "Revision number           : " << devProp.major << "." ;
	logger << devProp.minor << "\n" ;
	logger << "Name                      : " << devProp.name  << "\n" ;
	logger << "Total global memory       : " << globalMemoryInMB << " MB\n" ;
	logger << "Clock rate                : " << clockRateInGhz << " GHz\n" ;
	logger << "Number of multiprocessors : " << devProp.multiProcessorCount << "\n" ;
	logger << "Kernel execution timeout  : " ;
	logger << (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << "\n" ;

	logger << "\n" ;
}



void initializeGPU( int gpuId )
{
	if (gpuId >= 0)
	{
		CUDA_CHECK( cudaSetDevice(gpuId) ) ;
	}

	int i = -1 ;
	CUDA_CHECK( cudaGetDevice(&i) ) ;

	logger << "\nUsing CUDA device #" << i << "\n" ;

	cudaDeviceProp devProp;
	CUDA_CHECK(cudaGetDeviceProperties(&devProp, i));
	printDeviceProperties(devProp);
}



void finalize()
{
	cudaDeviceSynchronize() ;
	cudaProfilerStop() ;
	cudaDeviceReset() ;
}



void cudaCheck( cudaError_t cudaCode, std::string file, size_t line )
{
  if ( cudaSuccess != (cudaCode) )
  {
		std::stringstream sstr ;
		sstr << "Error at " << file << ":" << line 
    		 << " : " <<  cudaGetErrorString(cudaCode) << "\n" ;
    THROW (sstr.str()) ;
  }
}



}
