#include "gtest/gtest.h"
#include "BitSet.hpp"
#include "gpuTools.hpp"



using namespace microflow ;



TEST( BitSet, sizeof_CPU)
{
	ASSERT_EQ( 4u, sizeof(BitSet) ) ;
}



TEST( BitSet, test_CPU )
{
	ASSERT_NO_THROW
	(
		BitSet bitSet ;

		bitSet.clear() ;

		for (unsigned pos=0 ; pos < 32 ; pos++)
		{
			ASSERT_FALSE( bitSet.test(pos) ) ;
		}
	) ;
}



TEST( BitSet, set_CPU )
{
	ASSERT_NO_THROW
	(
		BitSet bitSet ;

		bitSet.clear() ;

		bitSet.set(0) ;
		bitSet.set(31) ;

		ASSERT_TRUE( bitSet.test(0) ) ;
		ASSERT_TRUE( bitSet.test(31) ) ;

		for (unsigned pos=1 ; pos < 31 ; pos++)
		{
			ASSERT_FALSE( bitSet.test(pos) ) ;
		}
	) ;
}



TEST( BitSet, outOfBounds_test_CPU )
{
	BitSet bitSet ;

	ASSERT_DEATH( bitSet.test(32), "" ) ;
}



TEST( BitSet, outOfBounds_set_CPU )
{
	BitSet bitSet ;

	ASSERT_DEATH( bitSet.set(32), "" ) ;
}



TEST( BitSet, operator_OR_CPU )
{
	BitSet bs1, bs2 ;

	bs1.clear() ; 
	bs2.clear() ;

	bs1.set(1) ;
	bs2.set(2) ;

	ASSERT_FALSE( bs1.test(2) ) ;
	ASSERT_FALSE( bs2.test(1) ) ;

	bs2 = bs2 | bs1 ;

	ASSERT_TRUE( bs2.test(1) ) ;
	ASSERT_TRUE( bs2.test(2) ) ;
	ASSERT_FALSE( bs2.test(0) ) ;
	for (unsigned pos=3 ; pos < 32 ; pos++)
	{
		ASSERT_FALSE( bs2.test(pos) ) ;
	}
}



TEST( BitSet, operator_EQ )
{
	BitSet bs1, bs2 ;

	bs1.clear() ; 
	bs2.clear() ;
	ASSERT_TRUE( bs1 == bs2 ) ;

	bs1.set(0) ;
	ASSERT_FALSE( bs1 == bs2 ) ;

	bs2.set(0) ;
	ASSERT_TRUE( bs1 == bs2 ) ;

	bs1.set(31) ;
	ASSERT_FALSE( bs1 == bs2 ) ;

	bs2.set(31) ;
	ASSERT_TRUE( bs1 == bs2 ) ;
}



TEST( BitSet, operator_out )
{
	BitSet bs ;
	std::stringstream ss ;

	bs.clear() ;
	ss << bs ;
	ASSERT_EQ( "00000000000000000000000000000000", ss.str() ) ;

	bs.set(31) ;
	ss.str("") ;
	ss << bs ;
	ASSERT_EQ( "00000000000000000000000000000001", ss.str() ) ;

	bs.clear() ;
	bs.set(0) ;
	ss.str("") ;
	ss << bs ;
	ASSERT_EQ( "10000000000000000000000000000000", ss.str() ) ;
}



static __device__ BitSet bitSetGPU ;



TEST( BitSet, sizeof_GPU )
{
	ASSERT_EQ( 4u, sizeof(bitSetGPU) ) ;
}



BitSet copyBitSetFromGPU()
{
	BitSet bitSet ;

	CUDA_CHECK(
			cudaMemcpyFromSymbol( &bitSet, bitSetGPU, 
														sizeof(bitSet), 0, cudaMemcpyDeviceToHost)
			) ;

	return bitSet ;
}



__global__ void 
kernelBitSet_Clear()
{
	for (unsigned pos=0 ; pos < 32 ; pos++)
	{
		bitSetGPU.set(pos) ;
	}

	bitSetGPU.clear() ;
}



TEST( BitSet, clear_GPU )
{
	dim3 numBlocks(1) ;
	dim3 numThreads(1,1,1) ;

	kernelBitSet_Clear<<< numBlocks, numThreads >>>() ;

	CUDA_CHECK( cudaPeekAtLastError() );
	CUDA_CHECK( cudaDeviceSynchronize() );  

	BitSet bitSet = copyBitSetFromGPU() ;

	for (unsigned pos=0 ; pos < 32 ; pos++)
	{
		ASSERT_FALSE( bitSet.test(pos) ) << "pos = " << pos << "\n" ;
	}
}



__global__ void 
kernelBitSet_Set( size_t pos )
{
	bitSetGPU.set( pos ) ;
}



TEST( BitSet, set_GPU )
{
	dim3 numBlocks(1) ;
	dim3 numThreads(1,1,1) ;

	kernelBitSet_Clear<<< numBlocks, numThreads >>>() ;

	kernelBitSet_Set<<< numBlocks, numThreads >>>(0) ;
	kernelBitSet_Set<<< numBlocks, numThreads >>>(31) ;
	kernelBitSet_Set<<< numBlocks, numThreads >>>(15) ;
	kernelBitSet_Set<<< numBlocks, numThreads >>>(16) ;

	CUDA_CHECK( cudaPeekAtLastError() );
	CUDA_CHECK( cudaDeviceSynchronize() );  

	BitSet bitSet = copyBitSetFromGPU() ;

	ASSERT_TRUE( bitSet.test(0) ) ;
	ASSERT_TRUE( bitSet.test(15) ) ;
	ASSERT_TRUE( bitSet.test(16) ) ;
	ASSERT_TRUE( bitSet.test(31) ) ;

	for (unsigned pos=1 ; pos < 15 ; pos++)
	{
		ASSERT_FALSE( bitSet.test(pos) ) << "pos = " << pos << "\n" ;
	}
	for (unsigned pos=17 ; pos < 31 ; pos++)
	{
		ASSERT_FALSE( bitSet.test(pos) ) << "pos = " << pos << "\n" ;
	}
}



static __device__ bool testResultGPU ;



bool copyTestResultFromGPU()
{
	bool testResult ;

	CUDA_CHECK(
			cudaMemcpyFromSymbol( &testResult, testResultGPU, 
														sizeof(testResult), 0, cudaMemcpyDeviceToHost)
			) ;

	return testResult ;
}

__global__ void 
kernelBitSet_Test( size_t pos )
{
	testResultGPU = bitSetGPU.test( pos ) ;
}



bool bitTest_testGPU (unsigned pos)
{
	dim3 numBlocks(1) ;
	dim3 numThreads(1,1,1) ;

	kernelBitSet_Test<<< numBlocks, numThreads >>>(pos) ;
	CUDA_CHECK( cudaPeekAtLastError() );
	CUDA_CHECK( cudaDeviceSynchronize() );  

	bool result = copyTestResultFromGPU() ;

	return result ;
}



TEST( BitSet, test_GPU )
{
	dim3 numBlocks(1) ;
	dim3 numThreads(1,1,1) ;

	kernelBitSet_Clear<<< numBlocks, numThreads >>>() ;

	for (unsigned pos=0 ; pos < 32 ; pos++)
	{
		ASSERT_FALSE( bitTest_testGPU(pos) ) ;
	}

	kernelBitSet_Set<<< numBlocks, numThreads >>>(0) ;
	kernelBitSet_Set<<< numBlocks, numThreads >>>(31) ;
	kernelBitSet_Set<<< numBlocks, numThreads >>>(15) ;
	kernelBitSet_Set<<< numBlocks, numThreads >>>(16) ;

	CUDA_CHECK( cudaPeekAtLastError() );
	CUDA_CHECK( cudaDeviceSynchronize() );  


	ASSERT_TRUE( bitTest_testGPU(0) ) ;
	ASSERT_TRUE( bitTest_testGPU(15) ) ;
	ASSERT_TRUE( bitTest_testGPU(16) ) ;
	ASSERT_TRUE( bitTest_testGPU(31) ) ;

	for (unsigned pos=1 ; pos < 15 ; pos++)
	{
		ASSERT_FALSE( bitTest_testGPU(pos) ) << "pos = " << pos << "\n" ;
	}
	for (unsigned pos=17 ; pos < 31 ; pos++)
	{
		ASSERT_FALSE( bitTest_testGPU(pos) ) << "pos = " << pos << "\n" ;
	}
}

