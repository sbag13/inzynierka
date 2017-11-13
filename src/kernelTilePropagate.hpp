#ifndef KERNEL_TILE_PROPAGATE_HPP
#define KERNEL_TILE_PROPAGATE_HPP



#include "TileDataArrangement.hpp"



/*
	Header for files with explicit template instantations used to speed up compilation.
*/



namespace microflow
{



class NodeType ;
class SolidNeighborMask ;
class PackedNodeNormalSet ;



template< 
					class LatticeArrangement,         
					class DataType,
					unsigned Edge,
					TileDataArrangement DataArrangement
				>
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
	) ;



}



#endif
