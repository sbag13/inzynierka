#ifndef TILE_TRAITS_COMMON_HH
#define TILE_TRAITS_COMMON_HH



#include <limits>



namespace microflow
{



#define TEMPLATE_TILE_TRAITS_COMMON                       \
template< unsigned NodesPerEdge, unsigned NDimensions >   \
INLINE



#define TILE_TRAITS_COMMON                      \
TileTraitsCommon< NodesPerEdge, NDimensions >



TEMPLATE_TILE_TRAITS_COMMON
constexpr unsigned TILE_TRAITS_COMMON::
getNNodesPerEdge()
{
	return NodesPerEdge ;
}



TEMPLATE_TILE_TRAITS_COMMON
constexpr unsigned TILE_TRAITS_COMMON::
getNNodesPerTile()
{
	return ( NDimensions == 2 ) ? NodesPerEdge * NodesPerEdge :
		   ( ( NDimensions == 3 ) ? NodesPerEdge * NodesPerEdge * NodesPerEdge : 
			 		(0u - 1u) ) ;
}



TEMPLATE_TILE_TRAITS_COMMON
constexpr unsigned TILE_TRAITS_COMMON::
computeNodeInTileIndex (unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ)
{
	return nodeInTileX +
				 nodeInTileY * getNNodesPerEdge() +
				 nodeInTileZ * getNNodesPerEdge() * getNNodesPerEdge() ;
}



TEMPLATE_TILE_TRAITS_COMMON
constexpr unsigned TILE_TRAITS_COMMON::
computeTileNodesBeginIndex (unsigned tileIndex)
{
	return tileIndex * getNNodesPerTile() ;
}



TEMPLATE_TILE_TRAITS_COMMON
constexpr unsigned TILE_TRAITS_COMMON::
computeNodeIndex (unsigned nodeInTileX, 
								  unsigned nodeInTileY,
								  unsigned nodeInTileZ,
								  unsigned tileIndex)
{
	return computeTileNodesBeginIndex (tileIndex) +
				 computeNodeInTileIndex (nodeInTileX, nodeInTileY, nodeInTileZ) ;
}



#undef TILE_TRAITS_COMMON
#undef TEMPLATE_TILE_TRAITS_COMMON



}



#endif
