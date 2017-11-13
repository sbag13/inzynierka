#include "kernelTilePropagate.tcc"



/*
	Explicit template instantation used to speed up compilation.
*/



namespace microflow
{



template
__global__ void 
kernelTilePropagateOpt
<
	LatticeArrangement<3,19>,
	double,
	4,
	TileDataArrangement::XYZ
>
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
		double            * __restrict__ tiledAllValues
	) ;


	
template
__global__ void 
kernelTilePropagateOpt
<
	LatticeArrangement<3,19>,
	double,
	4,
	TileDataArrangement::OPT_1
>
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
		double            * __restrict__ tiledAllValues
	) ;


	
}
