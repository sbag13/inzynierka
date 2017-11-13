#include "kernelTileGatherProcessBoundaryCollide.tcc"



/*
	Explicit template instantation used to speed up compilation.
*/



namespace microflow
{



template
__global__ void 
kernelTileGatherProcessBoundaryCollide
<
	FluidModelQuasicompressible,
	CollisionModelBGK,
	LatticeArrangement<3,19>,
	double,
	4,
	DataFlowDirection::FPOST_TO_F,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::XYZ>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::XYZ,
	SaveRhoU
>
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 double              * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 double rho0LB,
	 double u0LB_x,
	 double u0LB_y,
	 double u0LB_z,
	 double tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 double invRho0LB,
	 double invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



template
__global__ void 
kernelTileGatherProcessBoundaryCollide
<
	FluidModelQuasicompressible,
	CollisionModelBGK,
	LatticeArrangement<3,19>,
	double,
	4,
	DataFlowDirection::F_TO_FPOST,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::XYZ>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::XYZ,
	SaveRhoU
>
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 double              * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 double rho0LB,
	 double u0LB_x,
	 double u0LB_y,
	 double u0LB_z,
	 double tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 double invRho0LB,
	 double invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



template
__global__ void 
kernelTileGatherProcessBoundaryCollide
<
	FluidModelQuasicompressible,
	CollisionModelBGK,
	LatticeArrangement<3,19>,
	double,
	4,
	DataFlowDirection::FPOST_TO_F,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::OPT_1>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::OPT_1,
	SaveRhoU
>
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 double              * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 double rho0LB,
	 double u0LB_x,
	 double u0LB_y,
	 double u0LB_z,
	 double tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 double invRho0LB,
	 double invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



template
__global__ void 
kernelTileGatherProcessBoundaryCollide
<
	FluidModelQuasicompressible,
	CollisionModelBGK,
	LatticeArrangement<3,19>,
	double,
	4,
	DataFlowDirection::F_TO_FPOST,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::OPT_1>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::OPT_1,
	SaveRhoU
>
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 double              * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 double rho0LB,
	 double u0LB_x,
	 double u0LB_y,
	 double u0LB_z,
	 double tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 double invRho0LB,
	 double invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



template
__global__ void 
kernelTileGatherProcessBoundaryCollide
<
	FluidModelQuasicompressible,
	CollisionModelBGK,
	LatticeArrangement<3,19>,
	double,
	4,
	DataFlowDirection::FPOST_TO_F,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::XYZ>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::XYZ,
	DontSaveRhoU
>
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 double              * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 double rho0LB,
	 double u0LB_x,
	 double u0LB_y,
	 double u0LB_z,
	 double tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 double invRho0LB,
	 double invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



template
__global__ void 
kernelTileGatherProcessBoundaryCollide
<
	FluidModelQuasicompressible,
	CollisionModelBGK,
	LatticeArrangement<3,19>,
	double,
	4,
	DataFlowDirection::F_TO_FPOST,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::XYZ>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::XYZ,
	DontSaveRhoU
>
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 double              * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 double rho0LB,
	 double u0LB_x,
	 double u0LB_y,
	 double u0LB_z,
	 double tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 double invRho0LB,
	 double invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



template
__global__ void 
kernelTileGatherProcessBoundaryCollide
<
	FluidModelQuasicompressible,
	CollisionModelBGK,
	LatticeArrangement<3,19>,
	double,
	4,
	DataFlowDirection::FPOST_TO_F,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::OPT_1>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::OPT_1,
	DontSaveRhoU
>
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 double              * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 double rho0LB,
	 double u0LB_x,
	 double u0LB_y,
	 double u0LB_z,
	 double tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 double invRho0LB,
	 double invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



template
__global__ void 
kernelTileGatherProcessBoundaryCollide
<
	FluidModelQuasicompressible,
	CollisionModelBGK,
	LatticeArrangement<3,19>,
	double,
	4,
	DataFlowDirection::F_TO_FPOST,
	ThreadMapper< TiledLattice<D3Q19,double,StorageOnGPU,TileDataArrangement::OPT_1>, 
								SingleBlockPerTile, SingleThreadPerNode >,
	TileDataArrangement::OPT_1,
	DontSaveRhoU
>
	(
	 size_t widthInNodes,
	 size_t heightInNodes,
	 size_t depthInNodes,

	 unsigned int * __restrict__ tileMap,

	 size_t nOfNonEmptyTiles,

	 NodeType            * __restrict__ tiledNodeTypes,
	 SolidNeighborMask   * __restrict__ tiledSolidNeighborMasks,
	 PackedNodeNormalSet * __restrict__ tiledNodeNormals,
	 double              * __restrict__ tiledAllValues,

	 size_t * __restrict__ nonEmptyTilesX0,
	 size_t * __restrict__ nonEmptyTilesY0,
	 size_t * __restrict__ nonEmptyTilesZ0,

	 double rho0LB,
	 double u0LB_x,
	 double u0LB_y,
	 double u0LB_z,
	 double tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	 double invRho0LB,
	 double invTau,
#endif
	 const NodeType defaultExternalEdgePressureNode,
	 const NodeType defaultExternalCornerPressureNode		
	) ;



}
