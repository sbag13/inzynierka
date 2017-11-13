#include "ExpandedNodeLayout.hpp"
#include "LatticeArrangementD3Q27.hpp"



namespace microflow
{



ExpandedNodeLayout::
ExpandedNodeLayout( NodeLayout & nodeLayout )
: nodeLayout_(nodeLayout)
{
	nodeNormals_.resize( nodeLayout.getSize(), PackedNodeNormalSet() ) ;
	solidNeighborMasks_.resize( nodeLayout.getSize(), SolidNeighborMask() ) ;
}



void ExpandedNodeLayout::
rebuildBoundaryNodes (const Settings & settings)
{
	computeSolidNeighborMasks() ;
	computeNormalVectors() ;
	nodeLayout_.temporaryMarkUndefinedBoundaryNodesAndCovers() ;
	classifyNodesPlacedOnBoundary (settings) ;
	classifyPlacementForBoundaryNodes (settings) ;
}



// TODO: Use solidNeighborMask_ instead of direct reading nodeLayout_.
void ExpandedNodeLayout::
computeNormalVectors()
{
	Size size = getSize() ;

	size_t Nz = size.getDepth () - 1 ;
	size_t Ny = size.getHeight() - 1 ;
	size_t Nx = size.getWidth () - 1 ;

	// TOP and BOTTOM directions
	for (size_t h=0 ; h <= Nz ; h++)
		for (size_t j=0 ; j < Ny ; j++)
			for (size_t i=0 ; i < Nx ; i++)
			{
				size_t jd = j + 1 ;
				size_t id = i + 1 ;

				if (
						 isNodePlacedOnBoundary(i , j , h)  &&
						 isNodePlacedOnBoundary(i , jd, h)  &&
						 isNodePlacedOnBoundary(id, j , h)  &&
						 isNodePlacedOnBoundary(id, jd, h) 
					 )
				{
					// TOP
					size_t hd = h + 1 ;
					if ( ( hd <= Nz && 
									( nodeLayout_.getNodeType(i , j , hd).isSolid() ||
									  nodeLayout_.getNodeType(i , jd, hd).isSolid() ||
									  nodeLayout_.getNodeType(id, j , hd).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, hd).isSolid()
									 )
								) || hd > Nz )
					{
						nodeNormals_[ Coordinates(i , j , h) ].addNormalVector( Direction::TOP ) ;
						nodeNormals_[ Coordinates(id, j , h) ].addNormalVector( Direction::TOP ) ;
						nodeNormals_[ Coordinates(i , jd, h) ].addNormalVector( Direction::TOP ) ;
						nodeNormals_[ Coordinates(id, jd, h) ].addNormalVector( Direction::TOP ) ;
					}

					//BOTTOM
					hd = h - 1 ;
					if ( ( h >= 1 && 
									( nodeLayout_.getNodeType(i , j , hd).isSolid() ||
									  nodeLayout_.getNodeType(i , jd, hd).isSolid() ||
									  nodeLayout_.getNodeType(id, j , hd).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, hd).isSolid()
									 )
							  ) || h < 1 )
					{
						nodeNormals_[ Coordinates(i , j , h) ].addNormalVector( Direction::BOTTOM ) ;
						nodeNormals_[ Coordinates(id, j , h) ].addNormalVector( Direction::BOTTOM ) ;
						nodeNormals_[ Coordinates(i , jd, h) ].addNormalVector( Direction::BOTTOM ) ;
						nodeNormals_[ Coordinates(id, jd, h) ].addNormalVector( Direction::BOTTOM ) ;
					}

				}
			}


	// NORTH and SOUTH directions
	for (size_t h=0 ; h < Nz ; h++)
		for (size_t j=0 ; j <= Ny ; j++)
			for (size_t i=0 ; i < Nx ; i++)
			{
				size_t hd = h + 1 ;
				size_t id = i + 1 ;

				if (
						 isNodePlacedOnBoundary(i , j, h )  &&
						 isNodePlacedOnBoundary(i , j, hd)  &&
						 isNodePlacedOnBoundary(id, j, h )  &&
						 isNodePlacedOnBoundary(id, j, hd) 
					 )
				{
					//NORTH
					size_t jd = j + 1 ;
					if ( ( jd <= Ny && 
									( nodeLayout_.getNodeType(i , jd, h ).isSolid() ||
									  nodeLayout_.getNodeType(i , jd, hd).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, h ).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, hd).isSolid()
									)
							 ) || jd > Ny )
					{
						nodeNormals_[ Coordinates(i , j, h ) ].addNormalVector( Direction::NORTH ) ;
						nodeNormals_[ Coordinates(id, j, h ) ].addNormalVector( Direction::NORTH ) ;
						nodeNormals_[ Coordinates(i , j, hd) ].addNormalVector( Direction::NORTH ) ;
						nodeNormals_[ Coordinates(id, j, hd) ].addNormalVector( Direction::NORTH ) ;
					}

					//SOUTH
					jd = j - 1 ;
					if ( ( j >= 1 &&
									( nodeLayout_.getNodeType(i , jd, h ).isSolid() ||
									  nodeLayout_.getNodeType(i , jd, hd).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, h ).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, hd).isSolid()
									)
								) || j < 1 )
					{
						nodeNormals_[ Coordinates(i , j, h ) ].addNormalVector( Direction::SOUTH ) ;
						nodeNormals_[ Coordinates(id, j, h ) ].addNormalVector( Direction::SOUTH ) ;
						nodeNormals_[ Coordinates(i , j, hd) ].addNormalVector( Direction::SOUTH ) ;
						nodeNormals_[ Coordinates(id, j, hd) ].addNormalVector( Direction::SOUTH ) ;
					}
				}
			}

	//EAST and WEST directions
	for (size_t h=0 ; h < Nz ; h++)
		for (size_t j=0 ; j < Ny ; j++)
			for (size_t i=0 ; i <= Nx ; i++)
			{
				size_t hd = h + 1 ;
				size_t jd = j + 1 ;
				if (
						 isNodePlacedOnBoundary(i, j , h )  &&
						 isNodePlacedOnBoundary(i, j , hd)  &&
						 isNodePlacedOnBoundary(i, jd, h )  &&
						 isNodePlacedOnBoundary(i, jd, hd) 
					 )
				{
					//EAST
					size_t id = i + 1 ;
					if ( ( id <= Nx && 
									( nodeLayout_.getNodeType(id, j , h ).isSolid() ||
									  nodeLayout_.getNodeType(id, j , hd).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, h ).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, hd).isSolid()
									)
								) || id > Nx )
					{
						nodeNormals_[ Coordinates(i, j , h ) ].addNormalVector( Direction::EAST ) ;
						nodeNormals_[ Coordinates(i, jd, h ) ].addNormalVector( Direction::EAST ) ;
						nodeNormals_[ Coordinates(i, j , hd) ].addNormalVector( Direction::EAST ) ;
						nodeNormals_[ Coordinates(i, jd, hd) ].addNormalVector( Direction::EAST ) ;
					}

					//WEST
					id = i - 1 ;
					if ( ( i >= 1 && 
									( nodeLayout_.getNodeType(id, j , h ).isSolid() ||
									  nodeLayout_.getNodeType(id, j , hd).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, h ).isSolid() ||
									  nodeLayout_.getNodeType(id, jd, hd).isSolid()
									)
								) ||  i < 1 )
					{
						nodeNormals_[ Coordinates(i, j , h ) ].addNormalVector( Direction::WEST ) ;
						nodeNormals_[ Coordinates(i, jd, h ) ].addNormalVector( Direction::WEST ) ;
						nodeNormals_[ Coordinates(i, j , hd) ].addNormalVector( Direction::WEST ) ;
						nodeNormals_[ Coordinates(i, jd, hd) ].addNormalVector( Direction::WEST ) ;
					}
				}
			}

	// Classify node location
	for (size_t h=0 ; h <= Nz ; h++)
		for (size_t j=0 ; j <= Ny ; j++)
			for (size_t i=0 ; i <= Nx ; i++)
			{
				Coordinates coordinates(i,j,h) ;

				nodeNormals_[ coordinates ].setEdgeNodeType(
									PackedNodeNormalSet::EdgeNodeType::CONCAVE_EXTERNAL) ;

				unsigned normalVectorsCounter = nodeNormals_[ coordinates ].getNormalVectorsCounter() ;
				SolidNeighborMask solidNeighborMask = solidNeighborMasks_[ coordinates ] ;

				if ( 2 == normalVectorsCounter )
				{
					if ( solidNeighborMask.hasAllStraightNeighborsNonSolid() )
					{
						nodeNormals_[ coordinates ].setEdgeNodeType(
								PackedNodeNormalSet::EdgeNodeType::CONVEX_INTERNAL) ;
					}
					else 
					{
						Coordinates n1 = nodeNormals_[coordinates].getNormalVector(0) ;
						Coordinates n2 = nodeNormals_[coordinates].getNormalVector(1) ;
						Coordinates n3 = n1 + n2 ;
						if ( Coordinates(0,0,0) == n3)
						{
							nodeNormals_[ coordinates ].setEdgeNodeType(
									PackedNodeNormalSet::EdgeNodeType::PARALLEL_WALLS) ;
						}
					}
				}
				else if ( 3 == normalVectorsCounter )
				{
					if ( solidNeighborMask.hasAllStraightAndSlantingNeighborsNonSolid() )
					{
						nodeNormals_[ coordinates ].setEdgeNodeType(
								PackedNodeNormalSet::EdgeNodeType::CONVEX_INTERNAL) ;
					}
					else
					{
						Coordinates n1 = nodeNormals_[coordinates].getNormalVector(0) ;
						Coordinates n2 = nodeNormals_[coordinates].getNormalVector(1) ;
						Coordinates n3 = nodeNormals_[coordinates].getNormalVector(2) ;
						Coordinates s  = n1 + n2 + n3 ;

						unsigned sum = 0 ;
						for (Direction d : Direction::D3Q27)
						{
							long long bl = s.getX() * d.getX() +
														 s.getY() * d.getY() +
														 s.getZ() * d.getZ() ;

							if ( 0 == bl && solidNeighborMask.isNeighborSolid(d) &&
									 						solidNeighborMask.isNeighborSolid(d.computeInverse()) 
								 )
							{
								sum++ ;
							}
						}

						// Two parallel and contacting walls
						const Coordinates zero(0,0,0) ;
						if (
									zero == (n1 + n2) ||
									zero == (n1 + n3) ||
									zero == (n2 + n3)
							 )
						{
							sum++ ;
							nodeNormals_[ coordinates ].setEdgeNodeType(
									PackedNodeNormalSet::EdgeNodeType::PARALLEL_WALLS) ;
						}

						if (0 == sum)
						{
							nodeNormals_[ coordinates ].setEdgeNodeType(
									PackedNodeNormalSet::EdgeNodeType::CORNER) ;
						}
					}
				}
						 
			}

	for (size_t h=0 ; h <= Nz ; h++)
		for (size_t j=0 ; j <= Ny ; j++)
			for (size_t i=0 ; i <= Nx ; i++)
			{
				Coordinates c(i,j,h) ;
				nodeNormals_[c].calculateResultantNormalVector() ;
			}
}



void ExpandedNodeLayout::
computeSolidNeighborMasks()
{
	Size size = getSize() ;

	size_t Nz = size.getDepth () ;
	size_t Ny = size.getHeight() ;
	size_t Nx = size.getWidth () ;

	for (size_t h=0 ; h < Nz ; h++)
		for (size_t j=0 ; j < Ny ; j++)
			for (size_t i=0 ; i < Nx ; i++)
			{
				Coordinates nodeCoordinates(i,j,h) ;

				if (not nodeLayout_.getNodeType (nodeCoordinates).isSolid())
				{
					for (Direction::DirectionIndex q=1 ; q < 27 ; q++)
					{
						Direction neighborDirection = Direction::D3Q27[q-1] ;
						Coordinates neighborCoordinates = nodeCoordinates - neighborDirection ;

						if ( not size.areCoordinatesInLimits( neighborCoordinates ) )
						{
							solidNeighborMasks_[ nodeCoordinates ].markSolidNeighbor( q-1 ) ;
						}
						else
						{
							NodeType neighborNode = nodeLayout_.getNodeType( neighborCoordinates ) ;

							if ( neighborNode.isSolid() )
							{
								solidNeighborMasks_[ nodeCoordinates ].markSolidNeighbor( q-1 ) ;
							}
						}
					}
				}
			}
}



void ExpandedNodeLayout::
classifyNodesPlacedOnBoundary (const Settings & settings)
{
	Size size = getSize() ;

	size_t Nz = size.getDepth () ;
	size_t Ny = size.getHeight() ;
	size_t Nx = size.getWidth () ;

	for (size_t h=0 ; h < Nz ; h++)
		for (size_t j=0 ; j < Ny ; j++)
			for (size_t i=0 ; i < Nx ; i++)
			{
				Coordinates nodeCoordinates(i,j,h) ;

				if ( 
						nodeLayout_.getNodeType(nodeCoordinates) == NodeLayout::BOUNDARY_MARKER 
					 )
				{
					PackedNodeNormalSet normalVectors = getNormalVectors(nodeCoordinates) ;

					switch (normalVectors.getNormalVectorsCounter())
					{
						case 0:
							break ;

						case 1:	//FIXME: UNTESTED.
						{
							NodeType oldNode = 	nodeLayout_.getNodeType( i,j,h ) ;
							oldNode.setBaseType( settings.getDefaultWallNode().getBaseType() ) ;
							nodeLayout_.setNodeType( i,j,h, oldNode ) ;
						}
							break ;

						case 2:
							{
								NodeType newNodeType ;

								switch (normalVectors.getEdgeNodeType())
								{
									case PackedNodeNormalSet::EdgeNodeType::CONCAVE_EXTERNAL:
									{
										Direction n1 = normalVectors.getNormalVector(0) ;
										Direction n2 = normalVectors.getNormalVector(1) ;

										Coordinates n1Coords = nodeCoordinates - n1 ;
										Coordinates n2Coords = nodeCoordinates - n2 ;

										NodeType neighbour1 = nodeLayout_.getNodeType(n1Coords) ;
										NodeType neighbour2 = nodeLayout_.getNodeType(n2Coords) ;

										if ( neighbour1 == NodeBaseType::PRESSURE ||
												 neighbour2 == NodeBaseType::PRESSURE   )
										{
											newNodeType = settings.getDefaultExternalEdgePressureNode() ;
										}
										else
										{
											newNodeType = settings.getDefaultExternalEdgeNode() ;
										}
									}
									break ;

									case PackedNodeNormalSet::EdgeNodeType::CONVEX_INTERNAL:
									{
										newNodeType = settings.getDefaultInternalEdgeNode() ;
									}
									break ;

									default: //FIXME: UNTESTED.
									{
										newNodeType = settings.getDefaultNotIdentifiedNode() ;
									}
								}
								
								NodeType oldNode = 	nodeLayout_.getNodeType( i,j,h ) ;
								oldNode.setBaseType( newNodeType.getBaseType() ) ;
								oldNode.setPlacementModifier( newNodeType.getPlacementModifier() ) ;
								nodeLayout_.setNodeType( i,j,h, oldNode ) ;
							}
							break ;

						case 3:
							{
								NodeType newNodeType ;

								switch (normalVectors.getEdgeNodeType())
								{
									case PackedNodeNormalSet::EdgeNodeType::CONCAVE_EXTERNAL:
									{
										Direction n1 = normalVectors.getNormalVector(0) ;
										Direction n2 = normalVectors.getNormalVector(1) ;
										Direction n3 = normalVectors.getNormalVector(2) ;

										Coordinates n1Coords = nodeCoordinates - n1 - n2 ;
										Coordinates n2Coords = nodeCoordinates - n2 - n3 ;
										Coordinates n3Coords = nodeCoordinates - n3 - n1 ;

										NodeType neighbour1 = nodeLayout_.getNodeType(n1Coords) ;
										NodeType neighbour2 = nodeLayout_.getNodeType(n2Coords) ;
										NodeType neighbour3 = nodeLayout_.getNodeType(n3Coords) ;

										if ( neighbour1 == NodeBaseType::PRESSURE ||
												 neighbour2 == NodeBaseType::PRESSURE ||
												 neighbour3 == NodeBaseType::PRESSURE   )
										{
											newNodeType =  settings.getDefaultExternalCornerPressureNode() ;
										}
										else
										{
											newNodeType = settings.getDefaultExternalCornerNode() ;
										}
									}
									break ;

									case PackedNodeNormalSet::EdgeNodeType::CONVEX_INTERNAL: 
									{ //FIXME: UNTESTED.
										newNodeType = settings.getDefaultInternalCornerNode() ;
									}
									break ;

									case PackedNodeNormalSet::EdgeNodeType::CORNER:
									{
										newNodeType = settings.getDefaultEdgeToPerpendicularWallNode() ;
									}
									break ;

									default: //FIXME: UNTESTED.
									{
										newNodeType = settings.getDefaultNotIdentifiedNode() ;
									}
								}

								NodeType oldNode = 	nodeLayout_.getNodeType( i,j,h ) ;
								oldNode.setBaseType( newNodeType.getBaseType() ) ;
								oldNode.setPlacementModifier( newNodeType.getPlacementModifier() ) ;
								nodeLayout_.setNodeType( i,j,h, oldNode ) ;
							}
							break ;

						default: //FIXME: UNTESTED.
							{
							NodeType oldNode = 	nodeLayout_.getNodeType( i,j,h ) ;
							oldNode.setBaseType( settings.getDefaultNotIdentifiedNode().getBaseType() ) ;
							nodeLayout_.setNodeType( i,j,h, oldNode ) ;
							}
							break ;
					}
					
				}
			}
}



void ExpandedNodeLayout::
classifyPlacementForBoundaryNodes (const Settings & settings)
{
	Size size = getSize() ;

	size_t Nz = size.getDepth () ;
	size_t Ny = size.getHeight() ;
	size_t Nx = size.getWidth () ;

	for (size_t h=0 ; h < Nz ; h++)
		for (size_t j=0 ; j < Ny ; j++)
			for (size_t i=0 ; i < Nx ; i++)
			{
				Coordinates nodeCoordinates(i,j,h) ;
				NodeType node = nodeLayout_.getNodeType(nodeCoordinates) ;

				if ( (	
							NodeBaseType::VELOCITY   == node.getBaseType()  ||
							NodeBaseType::VELOCITY_0 == node.getBaseType()  ||
							NodeBaseType::PRESSURE   == node.getBaseType()
						 )   &&
						PlacementModifier::NONE == node.getPlacementModifier()  
					 )
				{
					// 1. Node has only a single solid neighbor on one of the straight directions.
					// 2. All other solid neighbors are on the same plane as the node from point 1.

					SolidNeighborMask solidNeighborMask = solidNeighborMasks_.getValue( nodeCoordinates ) ;
					Direction solidNeighborsPlaneDirection = solidNeighborMask.
																											computeDirectionOfPlaneWithAllSolidNeighbors() ;

					switch (solidNeighborsPlaneDirection.get())
					{
						case Direction::SOUTH:
							node.setPlacementModifier( PlacementModifier::NORTH ) ;
							break ;

						case Direction::NORTH:
							node.setPlacementModifier( PlacementModifier::SOUTH ) ;
							break ;

						case Direction::TOP:
							node.setPlacementModifier( PlacementModifier::BOTTOM ) ;
							break ;

						case Direction::BOTTOM:
							node.setPlacementModifier( PlacementModifier::TOP ) ;
							break ;

						case Direction::EAST:
							node.setPlacementModifier( PlacementModifier::WEST ) ;
							break ;

						case Direction::WEST:
							node.setPlacementModifier( PlacementModifier::EAST ) ;
							break ;

						default:
							node.setBaseType (settings.getDefaultNotIdentifiedNode().getBaseType() ) ;
							break ;
					}

					nodeLayout_.setNodeType( nodeCoordinates, node ) ;
				}
			}
}



}
