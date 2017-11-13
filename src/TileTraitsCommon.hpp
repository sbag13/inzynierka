#ifndef TILE_TRAITS_COMMON_HPP
#define TILE_TRAITS_COMMON_HPP



#include "Cuboid.hpp"
#include "Direction.hpp"
#include "cudaPrefix.hpp"



namespace microflow
{



template< unsigned NodesPerEdge, unsigned NDimensions >
class TileTraitsCommon
{
	public:

		HD static constexpr unsigned getNNodesPerEdge() ;
		HD static constexpr unsigned getNNodesPerTile() ;

		//TODO: maybe the below methods should be placed in Tile classes ?
		// For different arrangements of nodes in tile the indices may be computed
		// in different ways.
		HD static constexpr unsigned computeNodeInTileIndex (unsigned nodeInTileX, 
																												 unsigned nodeInTileY, 
																												 unsigned nodeInTileZ) ;

		HD static constexpr unsigned computeTileNodesBeginIndex (unsigned tileIndex) ;

		HD static constexpr unsigned computeNodeIndex (unsigned nodeInTileX, 
																									 unsigned nodeInTileY,
																									 unsigned nodeInTileZ,
																									 unsigned tileIndex) ;
} ;



}



#include "TileTraitsCommon.hh"



#endif

