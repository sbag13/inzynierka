#include "kernelTileProcessBoundary.tcc"



/*
	Explicit template instantation used to speed up compilation.
*/



namespace microflow
{



template
__global__ void
kernelTileProcessBoundaryOpt 
<
	FluidModelQuasicompressible,
	CollisionModelMRT,
	LatticeArrangement<3,19>,
	double,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::XYZ>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::XYZ
>
(
	size_t nOfNonEmptyTiles,

	NodeType            * __restrict__ tiledNodeTypes,
	SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	double            * __restrict__ tiledAllValues,

	double rho0LB,
	double u0LB_x,
	double u0LB_y,
	double u0LB_z,
	double tau,
	const NodeType defaultExternalEdgePressureNode,
	const NodeType defaultExternalCornerPressureNode
) ;



template
__global__ void
kernelTileProcessBoundaryOpt 
<
	FluidModelQuasicompressible,
	CollisionModelMRT,
	LatticeArrangement<3,19>,
	double,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::OPT_1>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::OPT_1
>
(
	size_t nOfNonEmptyTiles,

	NodeType            * __restrict__ tiledNodeTypes,
	SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	double            * __restrict__ tiledAllValues,

	double rho0LB,
	double u0LB_x,
	double u0LB_y,
	double u0LB_z,
	double tau,
	const NodeType defaultExternalEdgePressureNode,
	const NodeType defaultExternalCornerPressureNode
) ;



}
