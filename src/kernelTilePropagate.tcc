#ifndef KERNEL_TILE_PROPAGATE_TCC
#define KERNEL_TILE_PROPAGATE_TCC



#include "NodeFromTile.hpp"
#include "TiledLattice.hpp"
#include "LatticeArrangement.hpp"
#include "kernelTilePropagate.hpp"
#include "kernelTileGatherProcessBoundaryCollide.tcc"



namespace microflow
{



// Used for not important template parameters.
class Empty {} ; 
template <class,class> class Empty2 {} ;



//FIXME: Duplicated in gather(...).
//FIXME: WARNING - different than RS Streaming(). This version reads only F_POST and writes F.
//			 If the version completely compatible with RS is needed, restore previous code,
//			 but remember to change the way of index computations.
template< 
					class LatticeArrangement,         
					class DataType,
					unsigned Edge,
					TileDataArrangement DataArrangement>
__global__ void 
kernelTilePropagateOpt
(
		size_t widthInNodes,
		size_t heightInNodes,
		size_t depthInNodes,

		unsigned int * __restrict__ tileMap,


		size_t nOfNonEmptyTiles,
		size_t * __restrict__ nonEmptyTilesX0,
		size_t * __restrict__ nonEmptyTilesY0,
		size_t * __restrict__ nonEmptyTilesZ0,

		NodeType          * __restrict__ tiledNodeTypes,
		SolidNeighborMask * __restrict__ tiledSolidNeighborMasks,
		DataType          * __restrict__ tiledAllValues
	)
{
	constexpr DataFlowDirection FlowDirection = DataFlowDirection::FPOST_TO_F ;

	const unsigned nodeInTileX = threadIdx.x ;
	const unsigned nodeInTileY = threadIdx.y ;
	const unsigned nodeInTileZ = threadIdx.z ;
	const unsigned tileIndex = blockIdx.x ;

	typedef Tile <LatticeArrangement, DataType, Edge, StorageInKernel, 
								DataArrangement> TileType ; 
	typedef NodeFromTile <TileType,DataStorageMethod::COPY> NodeFromTileCopyType ;

	// nodeCopy is used only to keep local copy of f_i functions. All operations on memory
	// should be done with the use of nodePointers. Memory interface of nodeCopy may become
	// deprecated.
	auto nodeCopy = NodeFromTileCopyType
											(
												nodeInTileX, nodeInTileY, nodeInTileZ, tileIndex,
												tiledNodeTypes, tiledSolidNeighborMasks,
												NULL, tiledAllValues
											) ;

	auto nodePointers = NodeFromTile <TileType,DataStorageMethod::POINTERS>
											(
												nodeInTileX, nodeInTileY, nodeInTileZ, tileIndex,
												tiledNodeTypes, tiledSolidNeighborMasks,
												NULL, tiledAllValues
											) ;

		gatherF <LatticeArrangement,DataType,Edge,FlowDirection,DataArrangement> 
			(
			 nodeCopy,
			 widthInNodes, heightInNodes, depthInNodes,
			 tileMap,
			 nOfNonEmptyTiles,
			 nonEmptyTilesX0, nonEmptyTilesY0, nonEmptyTilesZ0,
			 tiledNodeTypes, tiledSolidNeighborMasks, tiledAllValues
			) ;

		#pragma unroll
		for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
		{
			if (DataFlowDirection::FPOST_TO_F == FlowDirection)
			{
				nodePointers.f (q) = nodeCopy.f (q) ;
			}
			else if (DataFlowDirection::F_TO_FPOST == FlowDirection)
			{
				nodePointers.fPost (q) = nodeCopy.f (q) ;
			}
		}
}


	
}



#endif
