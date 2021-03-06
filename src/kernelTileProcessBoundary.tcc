#ifndef KERNEL_PROCESS_BOUNDARY_TCC
#define KERNEL_PROCESS_BOUNDARY_TCC



#include "kernelTileProcessBoundary.hpp"

#include "FluidModels.hpp"
#include "CollisionModels.hpp"
#include "LBMOperatorChooser.hpp"
#include "LatticeArrangement.hpp"
#include "TiledLattice.hpp"
#include "ThreadMapper.hpp"
#include "NodeCalculator.hpp"



namespace microflow
{



template< 
					template<class LatticeArrangement, class DataType>  
														class FluidModel,                 
														class CollisionModel,
														class LatticeArrangement,         
														class DataType,
														class ThreadMapper,
														TileDataArrangement DataArrangement
				>
__global__ void 
kernelTileProcessBoundaryOpt
	(
		size_t nOfNonEmptyTiles,

		NodeType            * __restrict__ tiledNodeTypes,
		SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
		PackedNodeNormalSet * __restrict__ tiledNodeNormals,
		DataType            * __restrict__ tiledAllValues,

	 DataType rho0LB,
	 DataType u0LB_x,
	 DataType u0LB_y,
	 DataType u0LB_z,
	 DataType tau,
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode
	)
{
	DataType u0LB[3] ;
	u0LB[0] = u0LB_x ;
	u0LB[1] = u0LB_y ;
	u0LB[2] = u0LB_z ;

	typedef typename TiledLattice <LatticeArrangement, DataType, StorageInKernel, 
																 DataArrangement>::TileType TileType;
	typedef NodeFromTile <TileType, DataStorageMethod::POINTERS> NodeFromTileType ;

	NodeFromTileType
		node 
		(
			threadIdx.x, threadIdx.y, threadIdx.z,
			blockIdx.x,
			tiledNodeTypes, tiledSolidNeighborMasks, tiledNodeNormals, tiledAllValues
		) ;	

	NodeCalculator< FluidModel, CollisionModel, LatticeArrangement, DataType, StorageInKernel >
		nodeCalculator (rho0LB, u0LB, tau, 
									#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
										1.0/rho0LB, 1.0/tau, //FIXME: kernel parameters !!! 
									#endif
										defaultExternalEdgePressureNode,
										defaultExternalCornerPressureNode ) ;

#ifdef __CUDA_ARCH__
	nodeCalculator.template processBoundary<ThreadMapper, NodeFromTileType> (node) ;
#endif
}

	

}



#endif
