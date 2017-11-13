#ifndef KERNEL_TILE_COLLIDE_HPP
#define KERNEL_TILE_COLLIDE_HPP



/*
	Header for files with explicit template instantations used to speed up compilation.
*/



#include "NodeType.hpp"
#include "TileDataArrangement.hpp"



namespace microflow
{



class SolidNeighborMask ;
class PackedNodeNormalSet ;



template< 
					template<class LatticeArrangement, class DataType>  
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement,         
														class DataType,
														TileDataArrangement DataArrangement>
__global__ void 
kernelTileCollideOpt
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
	) ;



}



#endif