#ifndef NODE_LAYOUT_HH
#define NODE_LAYOUT_HH



namespace microflow
{



inline
const NodeType & NodeLayout::
getNodeType( size_t x, size_t y, size_t z) const 
{
	Coordinates c(x,y,z) ;
	return nodeTypes_.getValue( c ) ;
}



inline
const NodeType & NodeLayout::
getNodeType( Coordinates coordinates ) const
{
	return nodeTypes_.getValue( coordinates ) ;
}



inline
const NodeType NodeLayout::
safeGetNodeType( Coordinates coordinates ) const
{
	Size size = getSize() ;

	if ( size.areCoordinatesInLimits( coordinates ) )
	{
		return getNodeType( coordinates ) ;
	}
	else
	{
		return NodeBaseType::SOLID ;
	}
}



inline
void NodeLayout::
setNodeType( size_t x, size_t y, size_t z, NodeType nodeType)
{
	Coordinates c(x,y,z) ;
	setNodeType( c, nodeType ) ;
}



inline
void NodeLayout::
setNodeType( Coordinates coordinates, NodeType nodeType)
{
	nodeTypes_.setValue( coordinates, nodeType ) ;
}



inline
Size NodeLayout::
getSize() const
{
	return nodeTypes_.getSize() ;
}



inline
bool NodeLayout::
hasNodeSolidNeighbors( const Coordinates & coordinates ) const 
{
	// Test if node is on geometry edges
	if ( (0 == coordinates.getX()) || 
			 (0 == coordinates.getY()) ||
			 (0 == coordinates.getZ()) )
	{
		return true ;
	}
	
	Size size = getSize() ;

	if ( ( (size.getWidth ()-1) == coordinates.getX()) || 
			 ( (size.getHeight()-1) == coordinates.getY()) ||
			 ( (size.getDepth ()-1) == coordinates.getZ()) )
	{
		return true ;
	}

	// Check all neighbour nodes
	for (auto d : Direction::D3Q27)
	{
		Coordinates neighbourCoordinates = Coordinates(d) + coordinates ;

		if ( getNodeType( neighbourCoordinates ).isSolid() )
		{
			return true ;
		}
	}

	return false ;
}



inline
void NodeLayout::
temporaryMarkBoundaryNodes()
{
	for (size_t z=0 ; z < getSize().getDepth() ; z++)
		for (size_t y=0 ; y < getSize().getHeight() ; y++)
			for (size_t x=0 ; x < getSize().getWidth() ; x++)
			{
				Coordinates coordinates(x,y,z) ;

				if (isNodePlacedOnBoundary(x,y,z))
				{
					setNodeType( coordinates, BOUNDARY_MARKER) ;
				}
			}
}



inline
void NodeLayout::
temporaryMarkUndefinedBoundaryNodesAndCovers()
{
	for (size_t y=0 ; y < getSize().getHeight() ; y++)
		for (size_t x=0 ; x < getSize().getWidth() ; x++)
			for (size_t z=0 ; z < getSize().getDepth() ; 
												z += (getSize().getDepth() - 1) )
			{
				Coordinates coordinates(x,y,z) ;

				if ( isNodePlacedOnBoundary(x,y,z) )
				{
					setNodeType( coordinates, BOUNDARY_MARKER) ;
				}
			}

	for (size_t z=1 ; z < getSize().getDepth() - 1 ; z++)
		for (size_t y=0 ; y < getSize().getHeight() ; y++)
			for (size_t x=0 ; x < getSize().getWidth() ; x++)
			{
				Coordinates coordinates(x,y,z) ;

				if ( 
							getNodeType(coordinates).isFluid() &&
							isNodePlacedOnBoundary(x,y,z)
					 )
				{
					setNodeType( coordinates, BOUNDARY_MARKER) ;
				}
			}
}



inline
bool NodeLayout::
isNodePlacedOnBoundary(size_t x, size_t y, size_t z) const
{
	Coordinates coordinates(x,y,z) ;

	return					
		(not getNodeType (coordinates).isSolid())   &&
		hasNodeSolidNeighbors (coordinates) ;
}



inline
NodeLayout::Iterator NodeLayout::
begin() 
{ 
	return nodeTypes_.begin() ; 
}



inline
NodeLayout::ConstIterator NodeLayout::
begin() const 
{ 
	return nodeTypes_.begin() ;
}



inline
NodeLayout::Iterator NodeLayout::
end() 
{ 
	return nodeTypes_.end() ;
}



inline
NodeLayout::ConstIterator NodeLayout::
end() const 
{ 
	return nodeTypes_.end() ; 
}



inline
const BoundaryDefinitions & NodeLayout::
getBoundaryDefinitions() const
{
	return boundaryDefinitions_ ;
}



inline
void NodeLayout::
setBoundaryDefinitions (BoundaryDefinitions boundaryDefinitions)
{
	boundaryDefinitions_ = boundaryDefinitions ;
}



}



#endif
