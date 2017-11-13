#ifndef LATTICE_CALCULATOR_TCC
#define LATTICE_CALCULATOR_TCC



#ifdef __CUDACC__



#include "kernelTileGatherProcessBoundaryCollide.hpp"

#include "ThreadMapper.hpp"
#include "TiledLattice.hpp"

#include "kernelTileProcessBoundary.hpp"
#include "kernelTileCollide.hpp"
#include "kernelTilePropagate.hpp"
#include "gpuAlgorithms.hh"



namespace microflow
{



#define TEMPLATE_LATTICE_CALCULATOR                           \
template<                                                     \
					template<class LatticeArrangement, class DataType>  \
														class FluidModel,                 \
														class CollisionModel,             \
														class LatticeArrangement,         \
														class DataType,                   \
														TileDataArrangement DataArrangement>



#define LATTICE_CALCULATOR_GPU   \
LatticeCalculator <FluidModel, CollisionModel, LatticeArrangement, DataType, StorageOnGPU, \
									 DataArrangement>	



TEMPLATE_LATTICE_CALCULATOR
LATTICE_CALCULATOR_GPU::
LatticeCalculator (DataType rho0LB, 
									 DataType u0LB[ LatticeArrangement::getD()],
									 DataType tau,
									 NodeType defaultExternalEdgePressureNode,
									 NodeType defaultExternalCornerPressureNode 
									)
: CalculatorType( rho0LB, u0LB, tau
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	, 1.0/rho0LB, 1.0/tau
#endif
 )
, defaultExternalEdgePressureNode_ (defaultExternalEdgePressureNode)
, defaultExternalCornerPressureNode_ (defaultExternalCornerPressureNode)
{
}



template< 
														class Operator,
					template<class LatticeArrangement, class DataType>  
														class FluidModel,                 
														class CollisionModel,
														class LatticeArrangement,         
														class DataType,
														TileDataArrangement DataArrangement>
__global__ void 
kernelProcessTile
	(
		TiledLattice <LatticeArrangement, DataType, StorageInKernel, DataArrangement> tiledLattice,

	 DataType rho0LB,
	 DataType u0LB_x,
	 DataType u0LB_y,
	 DataType u0LB_z,
	 DataType tau,
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode
	)
{
	auto tile = tiledLattice.getTile( blockIdx.x ) ;

	DataType u0LB[3] ;
	u0LB[0] = u0LB_x ;
	u0LB[1] = u0LB_y ;
	u0LB[2] = u0LB_z ;

	auto node = tile.getNode( threadIdx.x, threadIdx.y, threadIdx.z ) ;
	NodeCalculator <FluidModel, CollisionModel, LatticeArrangement, DataType, StorageInKernel>
		nodeCalculator (rho0LB, u0LB, tau, 
									#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
										1.0/rho0LB, 1.0/tau, //FIXME: kernel parameters !!! 
									#endif
										defaultExternalEdgePressureNode,
										defaultExternalCornerPressureNode ) ;

	Operator::apply( nodeCalculator, node ) ;
}



TEMPLATE_LATTICE_CALCULATOR
template< class Operator >
void LATTICE_CALCULATOR_GPU::
processTiles( TiledLatticeType & tiledLattice )
{
	ThreadMapper< TiledLatticeType, SingleBlockPerTile, SingleThreadPerNode >
			threadMapper( tiledLattice ) ;

	dim3 numBlocks  = threadMapper.computeGridDimension() ;
	dim3 numThreads = threadMapper.computeBlockDimension() ;

	DataType u0LB_x = CalculatorType::u0LB_[0] ;
	DataType u0LB_y = CalculatorType::u0LB_[1] ;
	DataType u0LB_z = CalculatorType::u0LB_[2] ;

	// TODO: implicit conversion ?
	TiledLattice <LatticeArrangement, DataType, StorageInKernel, DataArrangement>
		tiledLatticeKernel (tiledLattice) ;
	
	kernelProcessTile <Operator,
										 FluidModel, CollisionModel, LatticeArrangement, DataType, DataArrangement>
															 <<<numBlocks, numThreads>>>
															 ( 
																tiledLatticeKernel,

																CalculatorType::rho0LB_, 
																u0LB_x, u0LB_y, u0LB_z,
																CalculatorType::tau_,
																defaultExternalEdgePressureNode_,
																defaultExternalCornerPressureNode_
															 ) ;

  CUDA_CHECK( cudaPeekAtLastError() );
  CUDA_CHECK( cudaDeviceSynchronize() );  	
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
initializeAtEquilibrium( TiledLatticeType & tiledLattice )
{
	processTiles< InitializatorAtEquilibrium >( tiledLattice ) ;

	tiledLattice.setValidCopyIDToF() ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
collideOpt (TiledLatticeType & tiledLattice)
{
	if (not tiledLattice.isValidCopyIDF())
	{
		THROW ("For collision valid data MUST be in F array") ;
	}

	ThreadMapper< TiledLatticeType, SingleBlockPerTile, SingleThreadPerNode >
			threadMapper( tiledLattice ) ;

	dim3 numBlocks  = threadMapper.computeGridDimension() ;
	dim3 numThreads = threadMapper.computeBlockDimension() ;

	DataType u0LB_x = CalculatorType::u0LB_[0] ;
	DataType u0LB_y = CalculatorType::u0LB_[1] ;
	DataType u0LB_z = CalculatorType::u0LB_[2] ;


	kernelTileCollideOpt <FluidModel, CollisionModel, LatticeArrangement, DataType,
												DataArrangement>
											 <<<numBlocks, numThreads>>>
		(
		 tiledLattice.getTileLayout().getNoNonEmptyTiles(),

		 tiledLattice.getNodeTypesPointer(),
		 tiledLattice.getSolidNeighborMasksPointer(),
		 tiledLattice.getNodeNormalsPointer(),
		 tiledLattice.getAllValuesPointer(),

		 CalculatorType::rho0LB_, 
		 u0LB_x, u0LB_y, u0LB_z,
		 CalculatorType::tau_,
		 defaultExternalEdgePressureNode_,
		 defaultExternalCornerPressureNode_
		) ;

  CUDA_CHECK( cudaPeekAtLastError() );
  CUDA_CHECK( cudaDeviceSynchronize() );  	

	tiledLattice.setValidCopyIDToFPost() ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
collide( TiledLatticeType & tiledLattice )
{
	collideOpt (tiledLattice) ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
processBoundaryOpt (TiledLatticeType & tiledLattice)
{
	if (not tiledLattice.isValidCopyIDF())
	{
		THROW ("For boundary processing valid data MUST be in F array") ;
	}

	typedef ThreadMapper< TiledLatticeType, SingleBlockPerTile, SingleThreadPerNode >
						ThreadMapperType ;

	ThreadMapperType threadMapper( tiledLattice ) ;

	dim3 numBlocks  = threadMapper.computeGridDimension() ;
	dim3 numThreads = threadMapper.computeBlockDimension() ;

	DataType u0LB_x = CalculatorType::u0LB_[0] ;
	DataType u0LB_y = CalculatorType::u0LB_[1] ;
	DataType u0LB_z = CalculatorType::u0LB_[2] ;

	kernelTileProcessBoundaryOpt <FluidModel, CollisionModel, LatticeArrangement, DataType,
																ThreadMapperType, DataArrangement>
															 <<<numBlocks, numThreads>>>
															 ( 
																 tiledLattice.getTileLayout().getNoNonEmptyTiles(),
                                 
																 tiledLattice.getNodeTypesPointer(),
																 tiledLattice.getSolidNeighborMasksPointer(),
																 tiledLattice.getNodeNormalsPointer(),
																 tiledLattice.getAllValuesPointer(),

																 CalculatorType::rho0LB_, 
																 u0LB_x, u0LB_y, u0LB_z,
																 CalculatorType::tau_,
																 defaultExternalEdgePressureNode_,
																 defaultExternalCornerPressureNode_
															 ) ;

  CUDA_CHECK( cudaPeekAtLastError() );
  CUDA_CHECK( cudaDeviceSynchronize() );  

	tiledLattice.setValidCopyIDToF() ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
processBoundary( TiledLatticeType & tiledLattice )
{
	processBoundaryOpt (tiledLattice) ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
propagate( TiledLatticeType & tiledLattice )
{
	propagateOpt (tiledLattice) ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
propagateOpt( TiledLatticeType & tiledLattice )
{
	if (not tiledLattice.isValidCopyIDFPost())
	{
		THROW ("For propagation valid data MUST be in FPost array") ;
	}

	ThreadMapper< TiledLatticeType, SingleBlockPerTile, SingleThreadPerNode >
			threadMapper( tiledLattice ) ;

	dim3 numBlocks  = threadMapper.computeGridDimension() ;
	dim3 numThreads = threadMapper.computeBlockDimension() ;

	Size sizeInNodes = tiledLattice.getTileLayout().getSize() ;


	kernelTilePropagateOpt< LatticeArrangement, DataType, 
											 		TiledLatticeType::getNNodesPerTileEdge(),
													DataArrangement
											 	>
															 <<<numBlocks, numThreads>>>
															 ( 
																 sizeInNodes.getWidth(),
																 sizeInNodes.getHeight(),
																 sizeInNodes.getDepth(),
																 tiledLattice.getTileLayout().getTileMapPointer(),

																 tiledLattice.getTileLayout().getNoNonEmptyTiles(),
																 tiledLattice.getTileLayout().getTilesX0Pointer(),
																 tiledLattice.getTileLayout().getTilesY0Pointer(),
																 tiledLattice.getTileLayout().getTilesZ0Pointer(),

																 tiledLattice.getNodeTypesPointer(),
																 tiledLattice.getSolidNeighborMasksPointer(),
																 tiledLattice.getAllValuesPointer()
															 ) ;

  CUDA_CHECK( cudaPeekAtLastError() );
  CUDA_CHECK( cudaDeviceSynchronize() );  	

	tiledLattice.setValidCopyIDToF() ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
initializeAtEquilibriumForGather (TiledLatticeType & tiledLattice)
{
	initializeAtEquilibrium (tiledLattice) ;
	collide (tiledLattice) ;
}



// WARNING: single call copmutes TWO steps.
// FIXME: replace two steps with some internal variable defining, which kernel version should 
//				be currently called.
TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
gatherProcessBoundaryCollide 
(
	TiledLatticeType & tiledLattice, bool shouldComputeRhoU
)
{
	if (tiledLattice.isValidCopyIDNone())
	{
		THROW ("Uninitialised tiledLattice") ;
	}

	if (shouldComputeRhoU)
	{
		if (tiledLattice.isValidCopyIDFPost())
		{
			callKernelTileGatherProcessBoundaryCollide 
				<DataFlowDirection::FPOST_TO_F, SaveRhoU> (tiledLattice) ;
		}
		else if (tiledLattice.isValidCopyIDF())
		{
			callKernelTileGatherProcessBoundaryCollide 
				<DataFlowDirection::F_TO_FPOST, SaveRhoU> (tiledLattice) ;
		}
	}
	else
	{
		if (tiledLattice.isValidCopyIDFPost())
		{
			callKernelTileGatherProcessBoundaryCollide 
				<DataFlowDirection::FPOST_TO_F, DontSaveRhoU> (tiledLattice) ;
		}
		else if (tiledLattice.isValidCopyIDF())
		{
			callKernelTileGatherProcessBoundaryCollide 
				<DataFlowDirection::F_TO_FPOST, DontSaveRhoU> (tiledLattice) ;
		}
	}

	tiledLattice.switchValidCopyID() ;
}



template< 
					template<class LatticeArrangement, class DataType>  
														class FluidModel,                 
														class CollisionModel,
														class LatticeArrangement,         
														class DataType,
														TileDataArrangement DataArrangement>
__global__ void 
kernelSwapFPostWithF
	(
		TiledLattice <LatticeArrangement, DataType, StorageInKernel, DataArrangement> tiledLattice
	)
{
	auto tile = tiledLattice.getTile( blockIdx.x ) ;
	auto node = tile.getNode( threadIdx.x, threadIdx.y, threadIdx.z ) ;

	if (node.nodeType().isSolid()) 
	{
		return ;
	}

	for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
	{
		swap (node.f(q), node.fPost(q)) ;
	}
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_GPU::
swapFPostWithF (TiledLatticeType & tiledLattice)
{
	ThreadMapper< TiledLatticeType, SingleBlockPerTile, SingleThreadPerNode >
			threadMapper( tiledLattice ) ;

	dim3 numBlocks  = threadMapper.computeGridDimension() ;
	dim3 numThreads = threadMapper.computeBlockDimension() ;

	// TODO: implicit conversion ?
	TiledLattice <LatticeArrangement, DataType, StorageInKernel, DataArrangement>
		tiledLatticeKernel (tiledLattice) ;
	
	kernelSwapFPostWithF <FluidModel, CollisionModel, LatticeArrangement, DataType>
															 <<<numBlocks, numThreads>>>
															 ( 
																tiledLatticeKernel
															 ) ;

  CUDA_CHECK( cudaPeekAtLastError() );
  CUDA_CHECK( cudaDeviceSynchronize() );  	

	tiledLattice.switchValidCopyID() ;
}



TEMPLATE_LATTICE_CALCULATOR
template <DataFlowDirection DataFlowDirection, class ShouldSaveRhoU>
void LATTICE_CALCULATOR_GPU::
callKernelTileGatherProcessBoundaryCollide
	(TiledLatticeType & tiledLattice)
{
	typedef ThreadMapper <TiledLatticeType, 
												SingleBlockPerTile, SingleThreadPerNode>
					ThreadMapperType ;

	ThreadMapperType threadMapper (tiledLattice) ;

	dim3 numBlocks  = threadMapper.computeGridDimension() ;
	dim3 numThreads = threadMapper.computeBlockDimension() ;

	Size sizeInNodes = tiledLattice.getTileLayout().getSize() ;

	DataType u0LB_x = CalculatorType::u0LB_[0] ;
	DataType u0LB_y = CalculatorType::u0LB_[1] ;
	DataType u0LB_z = CalculatorType::u0LB_[2] ;


	CUDA_CHECK (cudaDeviceSetCacheConfig (cudaFuncCachePreferShared)) ;
	CUDA_CHECK (cudaDeviceSetSharedMemConfig
								(cudaSharedMemBankSizeEightByte)) ;

	kernelTileGatherProcessBoundaryCollide 
	<
		FluidModel, CollisionModel, LatticeArrangement, DataType, 
		TiledLatticeType::getNNodesPerTileEdge(),
		DataFlowDirection, ThreadMapperType, DataArrangement, 
		ShouldSaveRhoU
	>
	 <<<numBlocks, numThreads>>>
	 ( 
		 sizeInNodes.getWidth(),
		 sizeInNodes.getHeight(),
		 sizeInNodes.getDepth(),

		 tiledLattice.getTileLayout().getTileMapPointer(),

		 tiledLattice.getTileLayout().getNoNonEmptyTiles(),

		 tiledLattice.getNodeTypesPointer(),
		 tiledLattice.getSolidNeighborMasksPointer(),
		 tiledLattice.getNodeNormalsPointer(),
		 tiledLattice.getAllValuesPointer(),
		 
		 tiledLattice.getTileLayout().getTilesX0Pointer(),
		 tiledLattice.getTileLayout().getTilesY0Pointer(),
		 tiledLattice.getTileLayout().getTilesZ0Pointer(),

		 CalculatorType::rho0LB_, 
		 u0LB_x, u0LB_y, u0LB_z,
		 CalculatorType::tau_,
	#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
		 CalculatorType::invRho0LB_,
		 CalculatorType::invTau_,
	#endif
		 defaultExternalEdgePressureNode_,
		 defaultExternalCornerPressureNode_
	 ) ;

  CUDA_CHECK( cudaPeekAtLastError() );
  CUDA_CHECK( cudaDeviceSynchronize() );  	
}



#undef TEMPLATE_LATTICE_CALCULATOR
#undef LATTICE_CALCULATOR_GPU



}



#endif // __CUDACC__



#endif
