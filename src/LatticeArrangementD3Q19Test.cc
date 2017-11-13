#include "gtest/gtest.h"
#include "LatticeArrangement.hpp"
#include "gpuTools.hpp"



#include <iostream>



using namespace microflow ;



#define DISABLE_COMPILER_WARNING(d) (void)d ;



TEST( LatticeArrangementD3Q19, numberOfC )
{
	unsigned counter = 0 ;
	
	for ( auto d : D3Q19::c)
	{
		counter ++ ;
		DISABLE_COMPILER_WARNING(d) ;
	}

	EXPECT_EQ( 19u, counter ) ;
}



TEST( LatticeArrangementD3Q19, getIndex )
{
	unsigned i=0 ;

	for ( auto d : D3Q19::c )
	{
		unsigned directionIndex = D3Q19::getIndex(d) ;

		EXPECT_EQ( d, D3Q19::c[ directionIndex ] ) 
			<< " i = " << i
			<< ", in direction : " << Direction(d) 
			<< ", index = " << directionIndex 
			<< ", out direction : " << Direction(D3Q19::c[ directionIndex ]) << "\n" ;

		i++ ;
	}
}



TEST( LatticeArrangementD3Q19, getName )
{
	std::cout << D3Q19::getName() << "\n" ;
	EXPECT_EQ( "D3Q19", D3Q19::getName() ) ;
}



static __device__ unsigned indexGPU ;



__global__ void
kernelGetIndex(Direction::D d)
{
	indexGPU = D3Q19::getIndex(d) ;
}



TEST( LatticeArrangementD3Q19, getIndex_GPU )
{
	unsigned i=0 ;

	for ( auto d : D3Q19::c )
	{
		unsigned index = D3Q19::getIndex(d) ;

		dim3 numBlocks(1) ;
		dim3 numThreads(1,1,1) ;

		kernelGetIndex<<< numBlocks, numThreads >>>( d ) ;

		CUDA_CHECK( cudaPeekAtLastError() );
		CUDA_CHECK( cudaDeviceSynchronize() );  
		
		unsigned indexFromGPU ;
		
		CUDA_CHECK(
				cudaMemcpyFromSymbol( &indexFromGPU, indexGPU, 
															sizeof(indexFromGPU), 0, cudaMemcpyDeviceToHost)
				) ;


		EXPECT_EQ( index, indexFromGPU ) 
			<< " i = " << i
			<< ", in direction : " << Direction(d) 
			<< ", index = " << index 
			<< ", indexGPU " << indexFromGPU << "\n" ;

		i++ ;
	}
}




static __device__ Direction::D cGPU ;



__global__ void
kernelGetC(Direction::DirectionIndex index)
{
	cGPU = D3Q19::getC( index ) ;
}



TEST( LatticeArrangementD3Q19, getC_GPU )
{
	for ( Direction::DirectionIndex q=0 ; q < D3Q19::getQ() ; q++ )
	{
		Direction::D c = D3Q19::getC( q ) ;

		dim3 numBlocks(1) ;
		dim3 numThreads(1,1,1) ;

		kernelGetC<<< numBlocks, numThreads >>>( q ) ;

		CUDA_CHECK( cudaPeekAtLastError() );
		CUDA_CHECK( cudaDeviceSynchronize() );  
		
		Direction::D cFromGPU ;
		
		CUDA_CHECK(
				cudaMemcpyFromSymbol(&cFromGPU, cGPU, sizeof(cFromGPU), 0, cudaMemcpyDeviceToHost)
				) ;


		EXPECT_EQ( c, cFromGPU ) 
			<< " q = " << q << ", c = " << c << ", cGPU " << cFromGPU << "\n" ;
	}
}




static __device__ double wGPU ;



__global__ void
kernelGetW(Direction::DirectionIndex index)
{
	wGPU = D3Q19::getW( index ) ;
}



TEST( LatticeArrangementD3Q19, getW_GPU )
{
	for ( Direction::DirectionIndex q=0 ; q < D3Q19::getQ() ; q++ )
	{
		double w = D3Q19::getW( q ) ;

		dim3 numBlocks(1) ;
		dim3 numThreads(1,1,1) ;

		kernelGetW<<< numBlocks, numThreads >>>( q ) ;

		CUDA_CHECK( cudaPeekAtLastError() );
		CUDA_CHECK( cudaDeviceSynchronize() );  
		
		double wFromGPU ;
		
		CUDA_CHECK(
				cudaMemcpyFromSymbol(&wFromGPU, wGPU, sizeof(wFromGPU), 0, cudaMemcpyDeviceToHost)
				) ;


		EXPECT_EQ( w, wFromGPU ) 
			<< " q = " << q << ", w = " << w << ", wGPU " << wFromGPU << "\n" ;
	}
}

