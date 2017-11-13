#ifndef EXPANDED_NODE_LAYOUT_HH
#define EXPANDED_NODE_LAYOUT_HH



namespace microflow
{



inline
Size ExpandedNodeLayout::
getSize() const
{
	return nodeNormals_.getSize() ; // TODO: maybe better nodeLayout_.getSize() ?
}



inline
const NodeLayout & ExpandedNodeLayout::
getNodeLayout() const
{
	return nodeLayout_ ;
}



inline
PackedNodeNormalSet ExpandedNodeLayout::
getNormalVectors(const Coordinates & coordinates) const
{
	return nodeNormals_.getValue( coordinates ) ;
}



inline
PackedNodeNormalSet ExpandedNodeLayout::
getNormalVectors( unsigned x, unsigned y, unsigned z) const
{
	Coordinates coordinates( x, y, z ) ;
	
	return getNormalVectors( coordinates ) ;
}



inline
SolidNeighborMask ExpandedNodeLayout::
getSolidNeighborMask( const Coordinates & coordinates ) const
{
	return solidNeighborMasks_.getValue( coordinates ) ;
}



inline
SolidNeighborMask ExpandedNodeLayout::
getSolidNeighborMask( unsigned x, unsigned y, unsigned z ) const
{
	Coordinates coordinates(x,y,z) ;

	return getSolidNeighborMask( coordinates ) ;
}



inline
bool ExpandedNodeLayout::
isNodePlacedOnBoundary (unsigned x, unsigned y, unsigned z) const
{
	return getSolidNeighborMask (x,y,z).hasSolidNeighbor() ;
}



}



#endif
