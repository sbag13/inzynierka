#ifndef KERNEL_TILE_GATHER_PROCESS_BOUNDARY_COLLIDE_HPP
#define KERNEL_TILE_GATHER_PROCESS_BOUNDARY_COLLIDE_HPP



/*
	Header for files with explicit template instantations used to speed up compilation.
*/



#include "NodeType.hpp"
#include "TileDataArrangement.hpp"
#include "DataFlowDirection.hpp"



namespace microflow
{



class SolidNeighborMask ;
class PackedNodeNormalSet ;



class SaveRhoU
{
	public:
		HD bool operator() () const { return true ; }
} ;



class DontSaveRhoU
{
	public:
		HD bool operator() () const { return false ; }
} ;



template< 
					template<class LatticeArrangement, class DataType>  
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement,         
														class DataType,
														unsigned Edge,
// Two versions needed, because there is only one common pointer to all data, thus 
// it is not possible to easy exchange f with fPost.
														DataFlowDirection DataFlowDirection,
														class ThreadMapper, // Needed for GPU version of processBoundaryBounceBack2()
														TileDataArrangement DataArrangement,
														class ShouldSaveRhoU
														>
__global__ void 
kernelTileGatherProcessBoundaryCollide
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 DataType            * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 DataType rho0LB,
	 DataType u0LB_x,
	 DataType u0LB_y,
	 DataType u0LB_z,
	 DataType tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 DataType invRho0LB,
	 DataType invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



}



#endif
