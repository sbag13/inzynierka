#ifndef KERNEL_TILE_GATHER_PROCESS_BOUNDARY_COLLIDE_HH
#define KERNEL_TILE_GATHER_PROCESS_BOUNDARY_COLLIDE_HH



#include "kernelTileGatherProcessBoundaryCollide.hpp"

#include "FluidModels.hpp"
#include "CollisionModels.hpp"
#include "LBMOperatorChooser.hpp"
#include "LatticeArrangement.hpp"
#include "TiledLattice.hpp"
#include "ThreadMapper.hpp"



namespace microflow
{


/*
	 If Edge = 4 then there are two warps 32 threads each.
	 The first warp (for threadIdx.z = {0,1}) has a valid copy of neighborTileIndex for
	 x = -1 and x = 1. 
	 The second warp (for threadIdx.z = {2,3}) has a valid copy of neighborTileIndex for
	 x = 0.
 */
__device__ INLINE
int computeNeighborTileX()
{
	int neighborTileX ;

	if (threadIdx.z < 2) // WARP 0  =>  x \in {-1,1}
	{
		neighborTileX = -1 + 2 * static_cast<int>(threadIdx.z) ;
	}
	else								 // WARP 1  =>  x = 0
	{
		neighborTileX = 0 ;
	}

	return neighborTileX ;
}



// WARNING: Warp level programming.
template <class Tile>
__device__ INLINE
void loadCopyOfTileMap
(
	unsigned int (&neighborTileMapCopy) [3][3][3],

	size_t widthInNodes,
	size_t heightInNodes,
	size_t depthInNodes,

	unsigned int * __restrict__ tileMap,

	size_t nOfNonEmptyTiles,
	size_t * __restrict__ nonEmptyTilesX0,
	size_t * __restrict__ nonEmptyTilesY0,
	size_t * __restrict__ nonEmptyTilesZ0,

	const unsigned currentTileIndex
) 
{
	/*
		WARNING: Warp level programming to avoid additional __syncthreads().
		WARNING: We assume, that there is no out-of-order execution.

		If Edge = 4 then there are two warps 32 threads each.
		The first warp (for threadIdx.z = {0,1}) has valid copy of neighborTileIndex for
		x = -1 and x = 1. 
		The second warp (for threadIdx.z = {2,3}) has valid copy of neighborTileIndex for
		x = 0.
	*/
	constexpr unsigned Edge = Tile::getNNodesPerEdge() ;
	
	const unsigned currentTileMapX = nonEmptyTilesX0 [currentTileIndex] / Edge ;
	const unsigned currentTileMapY = nonEmptyTilesY0 [currentTileIndex] / Edge ;
	const unsigned currentTileMapZ = nonEmptyTilesZ0 [currentTileIndex] / Edge ;

	// An order of threads within the same warp does not matter. Until we stay
	// within the same warp (which means the same range of neighborTileX), threads 
	// may be freely reordered.
	const int neighborTileDirectionX = computeNeighborTileX() ;
	const int neighborTileDirectionY = threadIdx.y - 1 ;
	const int neighborTileDirectionZ = threadIdx.x - 1 ;

	const unsigned neighborTileMapX = currentTileMapX + neighborTileDirectionX ;
	const unsigned neighborTileMapY = currentTileMapY + neighborTileDirectionY ;
	const unsigned neighborTileMapZ = currentTileMapZ + neighborTileDirectionZ ;

	const unsigned widthInTiles  = widthInNodes / Edge ;
	const unsigned heightInTiles = heightInNodes / Edge ;
	const unsigned depthInTiles  = depthInNodes / Edge ;

	if (3 > threadIdx.x  &&  3 > threadIdx.y  &&  3 > threadIdx.z)
	{
		unsigned neighborTileIndex = EMPTY_TILE ;

		if (
				neighborTileMapX < widthInTiles  &&
				neighborTileMapY < heightInTiles &&
				neighborTileMapZ < depthInTiles
			 )
		{
			const unsigned neighborTileIndexInMap = neighborTileMapX + 
																							neighborTileMapY * widthInTiles + 
																							neighborTileMapZ * widthInTiles * heightInTiles ;
			neighborTileIndex = tileMap [neighborTileIndexInMap] ;
		}

		const int neighborTileX = 1 + neighborTileDirectionX ;
		const int neighborTileY = 1 + neighborTileDirectionY ;
		const int neighborTileZ = 1 + neighborTileDirectionZ ;

		neighborTileMapCopy [neighborTileZ][neighborTileY][neighborTileX] = neighborTileIndex ;
	}
}



// WARNING: Warp level programming.
template <class Tile>
__device__ INLINE
void loadCopyOfNodeTypes
(
	NodeType::PackedDataType (& nodeTypesCopy) [6][6][6],

	const unsigned int (& neighborTileMapCopy) [3][3][3],
	NodeType * __restrict__ tiledNodeTypes,
	const unsigned currentTileIndex
)
{
	// Nodes from current tile.
	{
		const int nodeInTileX = threadIdx.x ;
		const int nodeInTileY = threadIdx.y ;
		const int nodeInTileZ = threadIdx.z ;

		const unsigned nodeIndex = Tile::computeNodeIndex
															 (
															  nodeInTileX, nodeInTileY, nodeInTileZ,
															  currentTileIndex
															 ) ;

		const int nodeCopyX = 1 + nodeInTileX ;
		const int nodeCopyY = 1 + nodeInTileY ;
		const int nodeCopyZ = 1 + nodeInTileZ ;

		nodeTypesCopy [nodeCopyZ][nodeCopyY][nodeCopyX] = tiledNodeTypes [nodeIndex].packedData() ;
	}


	constexpr unsigned Edge = Tile::getNNodesPerEdge() ;

	if (threadIdx.z < 2) // WARP 0  =>  x \in {-1,1}
	{		// In this warp we have valid values of neighborTileMapCopy for x = -1 and x = 1.
		

		// West (x = -1) and east (x = 1) planes from neighbor tiles - full half-warps (2 x 16)
		{
			const int neighborTileX = 1 + computeNeighborTileX() ;
			const int neighborTileY = 1 + 0 ;
			const int neighborTileZ = 1 + 0 ;

			const unsigned neighborTileIndex = 
				neighborTileMapCopy [neighborTileZ][neighborTileY][neighborTileX] ;

			NodeType::PackedDataType tmpNode = 0 ; // Solid node
				
			if (EMPTY_TILE != neighborTileIndex)
			{
				const int nodeInTileX = (Edge-1) - threadIdx.z * (Edge-1) ;
				const int nodeInTileY = threadIdx.y ;
				const int nodeInTileZ = threadIdx.x ;

				const unsigned neighborNodeIndex = Tile::computeNodeIndex
																					 (
																					  nodeInTileX, nodeInTileY, nodeInTileZ,
																					  neighborTileIndex
																					 ) ;

				tmpNode = tiledNodeTypes [neighborNodeIndex].packedData() ; 
			}

			const int nodeCopyX = threadIdx.z * (Edge + 1) ;
			const int nodeCopyY = 1 + threadIdx.y ;
			const int nodeCopyZ = 1 + threadIdx.x ;

			nodeTypesCopy [nodeCopyZ][nodeCopyY][nodeCopyX] = tmpNode ;
		}


		// 4-nodes edges from left (x = -1) and right (x = 1) planes - 
		//    full half-warps, 4 edges per warp.
		{
			unsigned neighborTileX = threadIdx.z * 2 ; // 0 or 2
			unsigned neighborTileY = 0 ;
			unsigned neighborTileZ = 0 ;
    	
			unsigned nodeInTileX = (Edge-1) - threadIdx.z * (Edge-1) ;
			unsigned nodeInTileY = 0 ; 
			unsigned nodeInTileZ = 0 ;
    	
			unsigned nodeCopyX = threadIdx.z * (Edge + 1) ;
			unsigned nodeCopyY = 0 ;
			unsigned nodeCopyZ = 0 ;
    	
			switch (threadIdx.y) // TODO: Should I replace this with arithmetic computations ?
			{
				case 0: // strip in front
					neighborTileY = 0 ; 
					neighborTileZ = 1 ;
					
					nodeInTileY = Edge-1 ; 
					nodeInTileZ = threadIdx.x ;
					
					nodeCopyY = 0 ; 
					nodeCopyZ = 1 + threadIdx.x ;
					break ;

				case 1: // strip on top
					neighborTileY = 1 ; 
					neighborTileZ = 2 ;
					
					nodeInTileY = threadIdx.x ; 
					nodeInTileZ = 0 ;
					
					nodeCopyY = 1 + threadIdx.x ; 
					nodeCopyZ = Edge+1 ;
					break ;

				case 2: // strip in back
					neighborTileY = 2 ; 
					neighborTileZ = 1 ;

					nodeInTileY = 0 ; 
					nodeInTileZ = threadIdx.x ;
					
					nodeCopyY = Edge+1 ; 
					nodeCopyZ = 1 + threadIdx.x ;
					break ;

				case 3: // strip on bottom
					neighborTileY = 1 ; 
					neighborTileZ = 0 ;

					nodeInTileY = threadIdx.x ; 
					nodeInTileZ = Edge-1 ;

					nodeCopyY = 1 + threadIdx.x ; 
					nodeCopyZ = 0 ;
					break ;
			}
    	
			const unsigned neighborTileIndex = 
				neighborTileMapCopy [neighborTileZ][neighborTileY][neighborTileX] ;

			NodeType::PackedDataType tmpNode = 0 ; // Solid node
    	
			if (EMPTY_TILE != neighborTileIndex)
			{
				const unsigned neighborNodeIndex = Tile::computeNodeIndex
																					 (
																					 	nodeInTileX, nodeInTileY, nodeInTileZ,
																						neighborTileIndex
																					 ) ;
				tmpNode = tiledNodeTypes [neighborNodeIndex].packedData() ;
			}

			nodeTypesCopy [nodeCopyZ][nodeCopyY][nodeCopyX] = tmpNode ;
		}


		// 8 vertices
		{
			if (threadIdx.x < 2  &&  threadIdx.y < 2)
			{
				const int neighborTileX = 1 + (-1) + 2 * threadIdx.x ;
				const int neighborTileY = 1 + (-1) + 2 * threadIdx.y ;
				const int neighborTileZ = 1 + (-1) + 2 * threadIdx.z ;

				const unsigned neighborTileIndex = 
					neighborTileMapCopy [neighborTileZ][neighborTileY][neighborTileX] ;

				NodeType::PackedDataType tmpNode = 0 ; // Solid node

				if (EMPTY_TILE != neighborTileIndex)
				{
					const int nodeInTileX = (Edge-1) - threadIdx.x * (Edge-1) ;
					const int nodeInTileY = (Edge-1) - threadIdx.y * (Edge-1) ;
					const int nodeInTileZ = (Edge-1) - threadIdx.z * (Edge-1) ;

					const unsigned neighborNodeIndex = Tile::computeNodeIndex
																						 (
																							nodeInTileX, nodeInTileY, nodeInTileZ,
																							neighborTileIndex
																						 ) ;

					tmpNode = tiledNodeTypes [neighborNodeIndex].packedData() ;
				}

				const int nodeCopyX = (Edge+1) * threadIdx.x ;
				const int nodeCopyY = (Edge+1) * threadIdx.y ;
				const int nodeCopyZ = (Edge+1) * threadIdx.z ;

				nodeTypesCopy [nodeCopyZ][nodeCopyY][nodeCopyX] = tmpNode ;
			}
		}

	}
	else // WARP 1 => threadIdx.z = {2,3}, look at computeNeighborTileX()
	{		 // In this warp we have valid values of neighborTileMapCopy for x = 0.
		

		// South (y = -1) and north (y = 1) planes from neighbor tiles - full half-warps (2 x 16)
		{
			const int neighborTileX = 1 + 0 ;
			const int neighborTileY = 1 + (-1) + 2 * (threadIdx.z - 2) ; // y = {-1, 1}
			const int neighborTileZ = 1 + 0 ;

			const unsigned neighborTileIndex = 
				neighborTileMapCopy [neighborTileZ][neighborTileY][neighborTileX] ;

			NodeType::PackedDataType tmpNode = 0 ; // Solid node
			
			if (EMPTY_TILE != neighborTileIndex)
			{
				const int nodeInTileX = threadIdx.x ;
				const int nodeInTileY = (Edge-1) - (threadIdx.z - 2) * (Edge-1) ;
				const int nodeInTileZ = threadIdx.y ;

				const unsigned neighborNodeIndex = Tile::computeNodeIndex
																					 (
																					 	nodeInTileX, nodeInTileY, nodeInTileZ,
																						neighborTileIndex
																					 ) ;

				tmpNode = tiledNodeTypes [neighborNodeIndex].packedData() ;
			}

			const int nodeCopyX = 1 + threadIdx.x ;
			const int nodeCopyY = (Edge+1) * (threadIdx.z - 2) ;
			const int nodeCopyZ = 1 + threadIdx.y ;
			
			nodeTypesCopy [nodeCopyZ][nodeCopyY][nodeCopyX] = tmpNode ;
		}


		// Top (z = 1) and bottom (z = -1) planes from neigbor tiles - full half-warps (2 x 16)
		{
			const int neighborTileX = 1 + 0 ;
			const int neighborTileY = 1 + 0 ;
			const int neighborTileZ = 1 + (-1) + 2 * (threadIdx.z - 2) ; // z = {-1, 1}
		
			const unsigned neighborTileIndex = 
				neighborTileMapCopy [neighborTileZ][neighborTileY][neighborTileX] ;
			
			NodeType::PackedDataType tmpNode = 0 ; // Solid node.
	
			if (EMPTY_TILE != neighborTileIndex)
			{
				const int nodeInTileX = threadIdx.x ;
				const int nodeInTileY = threadIdx.y ;
				const int nodeInTileZ = (Edge-1) - (threadIdx.z - 2) * (Edge-1) ;

				const unsigned neighborNodeIndex = Tile::computeNodeIndex
																					 (
																					 	nodeInTileX, nodeInTileY, nodeInTileZ,
																						neighborTileIndex
																					 ) ;

				tmpNode = tiledNodeTypes [neighborNodeIndex].packedData() ;
			}

			const int nodeCopyX = 1 + threadIdx.x ;
			const int nodeCopyY = 1 + threadIdx.y ;
			const int nodeCopyZ = (Edge+1) * (threadIdx.z - 2) ;

			nodeTypesCopy [nodeCopyZ][nodeCopyY][nodeCopyX] = tmpNode ;
		}


		// 4 parallel edges from west to east - 4 x 4 threads.
		if (threadIdx.y < 2) // remove threads with y = {2,3}, leave only threadsIdx.y = {0,1}.
		{
			const int neighborTileX = 1 + 0 ;
			const int neighborTileY = 1 + 2 * threadIdx.y - 1 ;
			const int neighborTileZ = 1 + 2 * (threadIdx.z - 2) - 1 ;

			const unsigned neighborTileIndex = 
				neighborTileMapCopy [neighborTileZ][neighborTileY][neighborTileX] ;
			
			NodeType::PackedDataType tmpNode = 0 ; // Solid node

			if (EMPTY_TILE != neighborTileIndex)
			{
				const int nodeInTileX = threadIdx.x ; // 0..3
				const int nodeInTileY = (Edge-1) - threadIdx.y * (Edge-1) ;
				const int nodeInTileZ = (Edge-1) - (threadIdx.z-2) * (Edge-1) ;

				const unsigned neighborNodeIndex = Tile::computeNodeIndex
																					 (
																					 	nodeInTileX, nodeInTileY, nodeInTileZ,
																						neighborTileIndex
																					 ) ;

				tmpNode = tiledNodeTypes [neighborNodeIndex].packedData() ;
			}
			const int nodeCopyX = 1 + threadIdx.x ;
			const int nodeCopyY = threadIdx.y * (Edge+1) ;
			const int nodeCopyZ = (threadIdx.z-2) * (Edge+1) ;

			nodeTypesCopy [nodeCopyZ][nodeCopyY][nodeCopyX] = tmpNode ;
		}
	}
}



HD INLINE 
constexpr bool shouldCopyF (DataStorageMethod dataStorageMethod)
{
	return (DataStorageMethod::COPY == dataStorageMethod ? true : false) ;
}


//FIXME: Copy of kernelTilePropagateOpt(...)
template
<
	class LatticeArrangement,         
	class DataType,
	unsigned Edge,
	DataFlowDirection DataFlowDirection,
	TileDataArrangement DataArrangement,
	DataStorageMethod DataStorageMethod
>
__device__ INLINE 
void
gatherF 
(
	NodeFromTile 
	<
		Tile <LatticeArrangement, DataType, Edge, StorageInKernel, DataArrangement>,
		DataStorageMethod
	> & node,

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
	typedef Tile <LatticeArrangement, DataType, Edge, StorageInKernel, DataArrangement> TileType ;

	const unsigned tileIndex = node.getTileIndex() ;

	__shared__ NodeType::PackedDataType nodeTypesCopy [Edge+2][Edge+2][Edge+2] ;
	__shared__ unsigned 								neighborTileIndexMapCopy [3][3][3] ;


	// WARNING - warp level programming.
	loadCopyOfTileMap <TileType>
	(
		neighborTileIndexMapCopy,

		widthInNodes, heightInNodes, depthInNodes,
		tileMap,
		nOfNonEmptyTiles,
		nonEmptyTilesX0, nonEmptyTilesY0, nonEmptyTilesZ0,
		tileIndex
	) ;
	// Here we have filled neighborTileIndexMapCopy, at least in current warp.

	// WARNING - warp level programming.
	loadCopyOfNodeTypes <TileType>
	(
		nodeTypesCopy,
		neighborTileIndexMapCopy,
		tiledNodeTypes,
		tileIndex
	) ;


	__syncthreads() ;


	const unsigned currentNodeInTileX = node.getNodeInTileX() ;
	const unsigned currentNodeInTileY = node.getNodeInTileY() ;
	const unsigned currentNodeInTileZ = node.getNodeInTileZ() ;

	node.nodeType().packedData() = nodeTypesCopy [currentNodeInTileZ + 1]
																							 [currentNodeInTileY + 1]
																							 [currentNodeInTileX + 1] ;

	if (not node.nodeType().isSolid())
	{
		auto sourceData = TileType::Data::F_POST ;
		if (DataFlowDirection::FPOST_TO_F == DataFlowDirection)
		{
			sourceData = TileType::Data::F_POST ;
		}
		else if (DataFlowDirection::F_TO_FPOST == DataFlowDirection)
		{
			sourceData = TileType::Data::F ;
		}


		// Gather f for q = 0.
		{
			Direction::DirectionIndex q = 0 ;

			unsigned sourceIndex = 0u-1u ;
			if (shouldCopyF (DataStorageMethod))
			{
				// WARNING: for BB2 nodes we should read old f, not fPost. However, this would
				//					require writes to fPost at collision for BB2 nodes, what result in
				//					race conditions. Thus, we assume, that for BB2 nodes f = fPost at
				//					LBM iteration begin.
				sourceIndex = TileType::computeNodeDataIndex 
											(
												currentNodeInTileX ,
												currentNodeInTileY ,
												currentNodeInTileZ ,
												tileIndex ,
												sourceData ,
												// After swap of f and fPost old value of f is in fPost.
												//TileType::Data::F_POST, // Data reads are only from fPost.
												q
											) ;
			}

			sourceIndex = TileType::computeNodeDataIndex 
				(
				 currentNodeInTileX ,
				 currentNodeInTileY ,
				 currentNodeInTileZ ,
				 tileIndex ,
				 sourceData ,
				 q
				) ;

			if ((unsigned)(-1) != sourceIndex)
			{
				node.f (q) = tiledAllValues [sourceIndex] ;
			}
		}
		#pragma unroll		
		for (Direction::DirectionIndex q=1 ; q < LatticeArrangement::getQ()-1 ; q += 2)
		{
			Direction::DirectionIndex q_1 = q ;
			Direction::DirectionIndex q_2 = q + 1 ;

			unsigned sourceIndex_1 = 0u-1u ;
			unsigned sourceIndex_2 = 0u-1u ;
			if (shouldCopyF (DataStorageMethod))
			{
				// WARNING: for BB2 nodes we should read old f, not fPost. However, this would
				//					require writes to fPost at collision for BB2 nodes, what result in
				//					race conditions. Thus, we assume, that for BB2 nodes f = fPost at
				//					LBM iteration begin.
				sourceIndex_1 = TileType::computeNodeDataIndex 
											(
												currentNodeInTileX ,
												currentNodeInTileY ,
												currentNodeInTileZ ,
												tileIndex ,
												sourceData ,
												q_1
											) ;
				sourceIndex_2 = TileType::computeNodeDataIndex 
											(
												currentNodeInTileX ,
												currentNodeInTileY ,
												currentNodeInTileZ ,
												tileIndex ,
												sourceData ,
												q_2
											) ;
			}


			Direction const dC_1 (LatticeArrangement::getC (q_1)) ;
			Direction const dC_2 (LatticeArrangement::getC (q_2)) ;

			// upwind
			int neighborNodeInTileX_1 = (int)(currentNodeInTileX) - dC_1.getX() ;
			int neighborNodeInTileY_1 = (int)(currentNodeInTileY) - dC_1.getY() ;
			int neighborNodeInTileZ_1 = (int)(currentNodeInTileZ) - dC_1.getZ() ;

			int neighborNodeInTileX_2 = (int)(currentNodeInTileX) - dC_2.getX() ;
			int neighborNodeInTileY_2 = (int)(currentNodeInTileY) - dC_2.getY() ;
			int neighborNodeInTileZ_2 = (int)(currentNodeInTileZ) - dC_2.getZ() ;

			const int neighborNodeCopyX_1 = neighborNodeInTileX_1 + 1 ;
			const int neighborNodeCopyY_1 = neighborNodeInTileY_1 + 1 ;
			const int neighborNodeCopyZ_1 = neighborNodeInTileZ_1 + 1 ;

			const int neighborNodeCopyX_2 = neighborNodeInTileX_2 + 1 ;
			const int neighborNodeCopyY_2 = neighborNodeInTileY_2 + 1 ;
			const int neighborNodeCopyZ_2 = neighborNodeInTileZ_2 + 1 ;

			NodeType neighborNodeType_1 ;
			NodeType neighborNodeType_2 ;
			neighborNodeType_1.packedData() = 
				nodeTypesCopy [neighborNodeCopyZ_1][neighborNodeCopyY_1][neighborNodeCopyX_1] ;
			neighborNodeType_2.packedData() = 
				nodeTypesCopy [neighborNodeCopyZ_2][neighborNodeCopyY_2][neighborNodeCopyX_2] ;

			
			unsigned indexInIndexMapCopy_1 = 13 ;
			unsigned indexInIndexMapCopy_2 = 13 ;
			unsigned indexOfTileWithNeighborNode_1 ;
			unsigned indexOfTileWithNeighborNode_2 ;
			unsigned sourceIndexTmp_1 ;
			unsigned sourceIndexTmp_2 ;

			if (not neighborNodeType_1.isSolid()  ||  not neighborNodeType_2.isSolid())
			{
				unsigned * neighborTileIndexMapCopyPtr = & (neighborTileIndexMapCopy[0][0][0]) ;

				// Check, if node coordinates are outside tile.
				if ( 1 == dC_1.getX()  &&  currentNodeInTileX ==      0        ) indexInIndexMapCopy_1 -= 1 ; // -1
				if (-1 == dC_1.getX()  &&  currentNodeInTileX == (int)(Edge-1) ) indexInIndexMapCopy_1 += 1 ; //  1
				if ( 1 == dC_1.getY()  &&  currentNodeInTileY ==      0        ) indexInIndexMapCopy_1 -= 3 ; // -1
				if (-1 == dC_1.getY()  &&  currentNodeInTileY == (int)(Edge-1) ) indexInIndexMapCopy_1 += 3 ; //  1
				if ( 1 == dC_1.getZ()  &&  currentNodeInTileZ ==      0        ) indexInIndexMapCopy_1 -= 9 ; // -1
				if (-1 == dC_1.getZ()  &&  currentNodeInTileZ == (int)(Edge-1) ) indexInIndexMapCopy_1 += 9 ; //  1

				indexOfTileWithNeighborNode_1 = neighborTileIndexMapCopyPtr [indexInIndexMapCopy_1] ;

				neighborNodeInTileX_1 %= Edge ;
				neighborNodeInTileY_1 %= Edge ;
				neighborNodeInTileZ_1 %= Edge ;

				// Check, if node coordinates are outside tile.
				if ( 1 == dC_2.getX()  &&  currentNodeInTileX ==      0        ) indexInIndexMapCopy_2 -= 1 ; // -1
				if (-1 == dC_2.getX()  &&  currentNodeInTileX == (int)(Edge-1) ) indexInIndexMapCopy_2 += 1 ; //  1
				if ( 1 == dC_2.getY()  &&  currentNodeInTileY ==      0        ) indexInIndexMapCopy_2 -= 3 ; // -1
				if (-1 == dC_2.getY()  &&  currentNodeInTileY == (int)(Edge-1) ) indexInIndexMapCopy_2 += 3 ; //  1
				if ( 1 == dC_2.getZ()  &&  currentNodeInTileZ ==      0        ) indexInIndexMapCopy_2 -= 9 ; // -1
				if (-1 == dC_2.getZ()  &&  currentNodeInTileZ == (int)(Edge-1) ) indexInIndexMapCopy_2 += 9 ; //  1

				indexOfTileWithNeighborNode_2 = neighborTileIndexMapCopyPtr [indexInIndexMapCopy_2] ;

				neighborNodeInTileX_2 %= Edge ;
				neighborNodeInTileY_2 %= Edge ;
				neighborNodeInTileZ_2 %= Edge ;

				sourceIndexTmp_1 = TileType::computeDataBlockInTileIndex (sourceData,q_1) +
													 TileType::computeIndexInFArray 
													 	(
															neighborNodeInTileX_1, 
															neighborNodeInTileY_1, 
															neighborNodeInTileZ_1, 
															q_1) ;
				sourceIndexTmp_2 = TileType::computeDataBlockInTileIndex (sourceData,q_2) +
													 TileType::computeIndexInFArray 
													 	(
															neighborNodeInTileX_2, 
															neighborNodeInTileY_2, 
															neighborNodeInTileZ_2, 
															q_2) ;

			}

			if (not neighborNodeType_1.isSolid())
			{
				sourceIndexTmp_1 += TileType::
														computeTileDataBeginIndex (indexOfTileWithNeighborNode_1) ;

				if (not neighborNodeType_1.isSolid()) sourceIndex_1 = sourceIndexTmp_1 ;
			}

			if (not neighborNodeType_2.isSolid())
			{
				sourceIndexTmp_2 += TileType::
														computeTileDataBeginIndex (indexOfTileWithNeighborNode_2) ;

				if (not neighborNodeType_2.isSolid()) sourceIndex_2 = sourceIndexTmp_2 ;
			}

			if ((unsigned)(-1) != sourceIndex_1)
			{
				node.f (q_1) = tiledAllValues [sourceIndex_1] ;
			}
			if ((unsigned)(-1) != sourceIndex_2)
			{
				node.f (q_2) = tiledAllValues [sourceIndex_2] ;
			}
		}

	}
}



template< 
					template<class LatticeArrangement, class DataType>  
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement,         
														class DataType,
														unsigned Edge,
														DataFlowDirection DataFlowDirection,
														class ThreadMapper,
														TileDataArrangement DataArrangement,

// FIXME: Unfinished. At the moment only twp combinations (all methods with and
//				without rho/u save) are supported and tested. Left for future use, maybe 
//				to enable/disable u and rho saving.
														class EnabledOperations
				>
__device__ void 
tileGatherProcessBoundaryCollide
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
	)
{
	typedef Tile <LatticeArrangement, DataType, Edge, StorageInKernel, DataArrangement> TileType ;

	const unsigned nodeInTileX = threadIdx.x ;
	const unsigned nodeInTileY = threadIdx.y ;
	const unsigned nodeInTileZ = threadIdx.z ;
	const unsigned tileIndex = blockIdx.x ;

	typedef NodeFromTile <TileType,DataStorageMethod::COPY> NodeFromTileCopyType ;

	// nodeCopy is used only to keep local copy of f_i functions. All operations on memory
	// should be done with the use of nodePointers. Memory interface of nodeCopy may become
	// deprecated.
	auto nodeCopy = NodeFromTileCopyType
											(
												nodeInTileX, nodeInTileY, nodeInTileZ, tileIndex,
												tiledNodeTypes, tiledSolidNeighborMasks,
												tiledNodeNormals, tiledAllValues
											) ;

	auto nodePointers = NodeFromTile <TileType,DataStorageMethod::POINTERS>
											(
												nodeInTileX, nodeInTileY, nodeInTileZ, tileIndex,
												tiledNodeTypes, tiledSolidNeighborMasks,
												tiledNodeNormals, tiledAllValues
											) ;

	/*
		WARNING: gather reads only fPost, thus for boundary nodes it is difficult to 
						 guarantee that value of f_i for "invalid" directions is the same as
						 in RS code. In tests always run processBoundary() after gather to 
						 compute f_i for lacking directions.
	*/

	if (EnabledOperations::canPropagate())
	{
		gatherF <LatticeArrangement,DataType,Edge,DataFlowDirection,DataArrangement> 
			(
			 nodeCopy,
			 widthInNodes, heightInNodes, depthInNodes,
			 tileMap,
			 nOfNonEmptyTiles,
			 nonEmptyTilesX0, nonEmptyTilesY0, nonEmptyTilesZ0,
			 tiledNodeTypes, tiledSolidNeighborMasks, tiledAllValues
			) ;
	}

	if (
			EnabledOperations::canSaveRhoU() ||
			EnabledOperations::canCollide () ||
			EnabledOperations::canProcessBoundary()
		 )
	{
		// For GTX TITAN we need 16 blocks of 64 threads each to achieve 50% occupancy. If 
		// shared memory has 48 KB, then for each block we can use 3 KB. The array with 
		// all velocity values requires 64 thread * 3 double * 8 B = 1536 B, thus we have
		// some reserve.
		// Two velocity values require 1024 B which gives no significant difference, because
		// current kernel version already uses 1056 B.
		//
		// __shared__ variable has the lifetime of the block.
		//
		__shared__ DataType uCopy [LatticeArrangement::getD()]
															[ThreadMapper::getBlockDimZ()]
															[ThreadMapper::getBlockDimY()]
															[ThreadMapper::getBlockDimX()] ;

		nodeCopy.registerSharedU (uCopy) ;
	}


	if (not nodeCopy.nodeType().isSolid())
	{
		if (
				EnabledOperations::canCollide()          || 
				EnabledOperations::canProcessBoundary()
				)
		{
#pragma unroll
			for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
			{
				nodeCopy.u (d) = 0 ;
			}
			nodeCopy.rho() = rho0LB ;

			//TODO: for velocity nodes the velocity should be read from boundary settings
			//			or computed from f_i. Check, which version is faster.
			if (NodeBaseType::VELOCITY == nodeCopy.nodeType().getBaseType())
			{
#pragma unroll
				for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
				{
					nodeCopy.u (d) = nodePointers.u (d) ;
					//nodeCopy.u (d) = nodePointers.uBoundary (d) ; //FIXME: use this !
				}
			}

			DataType u0LB[3] ;
			u0LB[0] = u0LB_x ;
			u0LB[1] = u0LB_y ;
			u0LB[2] = u0LB_z ;

			NodeCalculator< FluidModel, CollisionModel, LatticeArrangement, DataType, StorageInKernel >
				nodeCalculator (rho0LB, u0LB, tau,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
						invRho0LB, invTau,
#endif
						defaultExternalEdgePressureNode,
						defaultExternalCornerPressureNode ) ;


			if (EnabledOperations::canProcessBoundary())
			{
			#ifdef __CUDA_ARCH__
				nodeCalculator.template processBoundary<ThreadMapper,NodeFromTileCopyType> (nodeCopy) ;
			#endif
				//FIXME: Restore BB2 nodes processing here - may improve performance.
			}

			
			if (EnabledOperations::canCollide())
			{
				nodeCalculator.template collide 
					<
						NodeFromTileCopyType,
						// Do not store to fPost, in parallel implementation based on gather fPost is
						// only read.
						WhereSaveF::F 
					> (nodeCopy) ;
			}

		}


		#pragma unroll
		for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
		{
			if (DataFlowDirection::FPOST_TO_F == DataFlowDirection)
			{
				nodePointers.f (q) = nodeCopy.f (q) ;   // This line adds two registers.
			}
			else if (DataFlowDirection::F_TO_FPOST == DataFlowDirection)
			{
				nodePointers.fPost (q) = nodeCopy.f (q) ;   // This line adds two registers.
			}
		}


		if (EnabledOperations::canSaveRhoU())
		{
			if (
					NodeBaseType::BOUNCE_BACK_2 != nodeCopy.nodeType().getBaseType()
				 )
			{
				#pragma unroll
				for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
				{
					nodePointers.u (d) = nodeCopy.u (d) ;
				}
				nodePointers.rho() = nodeCopy.rho() ;
			}
		}

	}
}



template< 
					template<class LatticeArrangement, class DataType>  
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement,         
														class DataType,
														unsigned Edge,
														DataFlowDirection DataFlowDirection,
														class ThreadMapper,
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
	)
{
	if (ShouldSaveRhoU()())
	{
		class EnabledOperations
		{
			public:
				HD static constexpr bool canPropagate      () { return true ; }
				HD static constexpr bool canCollide        () { return true ; }
				HD static constexpr bool canProcessBoundary() { return true ; }
				HD static constexpr bool canSaveRhoU       () { return true ; }
		} ;

		tileGatherProcessBoundaryCollide <FluidModel, CollisionModel, LatticeArrangement,
																			DataType, Edge, 
																			DataFlowDirection, ThreadMapper, DataArrangement, 
																			EnabledOperations>
																		 (
																			widthInNodes, heightInNodes, depthInNodes,
																			tileMap,
																			nOfNonEmptyTiles,
																			tiledNodeTypes, tiledSolidNeighborMasks, 
																			tiledNodeNormals, tiledAllValues,
																			nonEmptyTilesX0, nonEmptyTilesY0, nonEmptyTilesZ0,
																			rho0LB, u0LB_x, u0LB_y, u0LB_z,
																			tau,
																		#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
																			invRho0LB, invTau,
																		#endif
																			defaultExternalEdgePressureNode,
																			defaultExternalCornerPressureNode
																		 )	;
	}
	else
	{
		class EnabledOperations
		{
			public:
				HD static constexpr bool canPropagate      () { return true ; }
				HD static constexpr bool canCollide        () { return true ; }
				HD static constexpr bool canProcessBoundary() { return true ; }
				HD static constexpr bool canSaveRhoU       () { return false ; }
		} ;

		tileGatherProcessBoundaryCollide <FluidModel, CollisionModel, LatticeArrangement,
																			DataType, Edge, 
																			DataFlowDirection, ThreadMapper, DataArrangement, 
																			EnabledOperations>
																		 (
																			widthInNodes, heightInNodes, depthInNodes,
																			tileMap,
																			nOfNonEmptyTiles,
																			tiledNodeTypes, tiledSolidNeighborMasks, 
																			tiledNodeNormals, tiledAllValues,
																			nonEmptyTilesX0, nonEmptyTilesY0, nonEmptyTilesZ0,
																			rho0LB, u0LB_x, u0LB_y, u0LB_z,
																			tau,
																		#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
																			invRho0LB, invTau,
																		#endif
																			defaultExternalEdgePressureNode,
																			defaultExternalCornerPressureNode
																		 )	;
	}

}



}



#endif
