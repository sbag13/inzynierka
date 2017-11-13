#ifndef CUDA_PREFIX_HPP
#define CUDA_PREFIX_HPP



#ifdef __CUDACC__
	
	#define DEVICE __device__
	#define HOST __host__

	#define HD __host__ __device__

	#define HD_WARNING_DISABLE #pragma hd_warning_disable 

	#define INLINE __forceinline__

#else

	#define DEVICE
	#define HOST
	#define HD
	#define HD_WARNING_DISABLE
	#define HD_WARNING_ENABLE
	#define INLINE inline

#endif



#ifdef __CUDA_ARCH__

	#define CONSTEXPR

#else

	#define CONSTEXPR constexpr 

#endif



#endif
