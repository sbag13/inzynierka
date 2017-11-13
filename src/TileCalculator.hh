#ifndef TILE_CALCULATOR_HH
#define TILE_CALCULATOR_HH



#include "LBMOperatorChooser.hpp"



namespace microflow
{


#define TEMPLATE_TILE_CALCULATOR                              \
template<                                                     \
					template<class LatticeArrangement, class DataType>  \
														class FluidModel,                 \
														class CollisionModel,             \
														class LatticeArrangement,         \
														class DataType,                   \
														unsigned Edge,										\
														TileDataArrangement DataArrangement >


#define TILE_CALCULATOR_CPU			\
TileCalculator <FluidModel, CollisionModel, LatticeArrangement, DataType, Edge, StorageOnCPU, \
								DataArrangement>



TEMPLATE_TILE_CALCULATOR
TILE_CALCULATOR_CPU::
TileCalculator (DataType rho0LB, 
								DataType u0LB[LatticeArrangement::getD()],
								DataType tau, 
								NodeType defaultExternalEdgePressureNode,
								NodeType defaultExternalCornerPressureNode 
							 )
: CalculatorType( rho0LB, u0LB, tau
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	, 1.0/rho0LB, 1.0/tau
#endif
)
, defaultExternalEdgePressureNode_ (defaultExternalEdgePressureNode)
, defaultExternalCornerPressureNode_ (defaultExternalCornerPressureNode)
{
}



TEMPLATE_TILE_CALCULATOR
void TILE_CALCULATOR_CPU::
initializeAtEquilibrium( TileType & tile )
{
	processNodes< InitializatorAtEquilibrium >( tile ) ;
}



TEMPLATE_TILE_CALCULATOR
void TILE_CALCULATOR_CPU::
collide( TileType & tile )
{
	processNodes< Collider >( tile ) ;
}



TEMPLATE_TILE_CALCULATOR
void TILE_CALCULATOR_CPU::
processBoundary( TileType & tile )
{
	processNodes< BoundaryProcessor >( tile ) ;
}



TEMPLATE_TILE_CALCULATOR
void TILE_CALCULATOR_CPU::
computeRhoForBB2Nodes (TileType & tile)
{
	constexpr int edge = Edge ;
	for (int z=0 ; z < edge ; z++)
		for (int y=0 ; y < edge ; y++)
			for (int x=0 ; x < edge ; x++)
			{
				auto node = tile.getNode (x,y,z) ;

				if (node.nodeType() != NodeBaseType::BOUNCE_BACK_2)
				{
					continue ;
				}

				DataType vTmp = 0.0 ;
				DataType counter = 0.0 ;

				for (Direction::DirectionIndex q=1 ;
						 q < LatticeArrangement::getQ() ; q++)
				{
					Direction direction (LatticeArrangement::getC (q)) ;

					if (node.solidNeighborMask().isNeighborSolid (direction))
					{
						continue ;
					}

					//TODO: Maybe we should add method getNeighbor() to NodeFromTile ?
					int nx = x - direction.getX() ;
					int ny = y - direction.getY() ;
					int nz = z - direction.getZ() ;

					Direction neighborTileDirection (O) ;

					if (nx >= edge) neighborTileDirection.setX ( 1) ;
					if (nx <  0   ) neighborTileDirection.setX (-1) ;
					if (ny >= edge) neighborTileDirection.setY ( 1) ;
					if (ny <  0   ) neighborTileDirection.setY (-1) ;
					if (nz >= edge) neighborTileDirection.setZ ( 1) ;
					if (nz <  0   ) neighborTileDirection.setZ (-1) ;

					auto neighborTile = tile.getNeighbor (neighborTileDirection) ;

					if (neighborTile.isEmpty()) // Maybe not needed.
					{
						continue ;
					}

					nx %= static_cast<unsigned>(edge) ;
					ny %= static_cast<unsigned>(edge) ;
					nz %= static_cast<unsigned>(edge) ;

					auto neighborNode = neighborTile.getNode (nx,ny,nz) ;

					if (neighborNode.nodeType().isSolid() ||
							neighborNode.nodeType() == NodeBaseType::BOUNCE_BACK_2)
					{
						continue ;
					}

					vTmp += neighborNode.rho() ;
					counter += 1.0 ;
				}

				node.rho() = vTmp / counter ;
			}
}



TEMPLATE_TILE_CALCULATOR
template< class Operator >
void TILE_CALCULATOR_CPU::
processNodes( TileType & tile )
{
	for (unsigned z=0 ; z < Edge ; z++)
		for (unsigned y=0 ; y < Edge ; y++)
			for (unsigned x=0 ; x < Edge ; x++)
			{
				NodeCalculator< FluidModel, CollisionModel, LatticeArrangement, DataType, StorageOnCPU >
					calculator( CalculatorType::rho0LB_, 
											CalculatorType::u0LB_, 
											CalculatorType::tau_,
										#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
											CalculatorType::invRho0LB_,
											CalculatorType::invTau_,
									  #endif
											defaultExternalEdgePressureNode_,
											defaultExternalCornerPressureNode_ ) ;
				auto node = tile.getNode( x,y,z ) ;

				Operator::apply( calculator, node ) ;
			}
}



template< class TileType >
void propagateNodeAtTileBorder( TileType & tile,
																Coordinates currentNodeCoordinates,
																Direction neighborTileDirection )
{
	const unsigned edge = TileType::getNNodesPerEdge() ;
	const Size tileSize(edge, edge, edge) ;
	
	unsigned x = currentNodeCoordinates.getX() ;
	unsigned y = currentNodeCoordinates.getY() ;
	unsigned z = currentNodeCoordinates.getZ() ;

	auto currentNode = tile.getNode (x,y,z) ;

	NodeType nodeType = currentNode.nodeType() ;

	if ( nodeType.isSolid() ) return ;

	auto neighborTile = tile.getNeighbor( neighborTileDirection ) ;

	if ( neighborTile.isEmpty() ) return ;

	for ( 
				Direction::DirectionIndex q=0 ; 
				q < TileType::LatticeArrangementType::getQ() ;
				q++
			)
	{
		Direction c = TileType::LatticeArrangementType::getC( q ) ;

		// Find directions
		if ( (0 != neighborTileDirection.getX()) && 
				(neighborTileDirection.getX() != (- c.getX()) ) ) continue ;
		if ( (0 != neighborTileDirection.getY()) && 
				(neighborTileDirection.getY() != (- c.getY()) ) ) continue ;
		if ( (0 != neighborTileDirection.getZ()) && 
				(neighborTileDirection.getZ() != (- c.getZ()) ) ) continue ;

		unsigned nx = currentNodeCoordinates.getX() - c.getX() ;
		unsigned ny = currentNodeCoordinates.getY() - c.getY() ;
		unsigned nz = currentNodeCoordinates.getZ() - c.getZ() ;

		if ( 0 != neighborTileDirection.getX() )
		{
			nx %= edge ;
		}
		if ( 0 != neighborTileDirection.getY() )
		{
			ny %= edge ;
		}
		if ( 0 != neighborTileDirection.getZ() )
		{
			nz %= edge ;
		}

		Coordinates neighborNodeCoordinates( nx,ny,nz ) ; 

		if ( tileSize.areCoordinatesInLimits( neighborNodeCoordinates ) )
		{
			auto neighborNode = neighborTile.getNode (nx,ny,nz) ;

			if ( not neighborNode.nodeType().isSolid() )
			{
				currentNode.f (q) = neighborNode.fPost (q) ;
			}
		}
	}
}


TEMPLATE_TILE_CALCULATOR
void TILE_CALCULATOR_CPU::
propagateExternalNodes( TileType & tile, Cuboid wall, Direction neighborTileDirection )
{
	auto neighborTile = tile.getNeighbor( neighborTileDirection ) ;
	const Size tileSize(Edge, Edge, Edge) ;

	if ( not neighborTile.isEmpty() )
	{
		for (unsigned z = wall.zMin ; z <= wall.zMax ; z++)
			for (unsigned y = wall.yMin ; y <= wall.yMax ; y++)
				for (unsigned x = wall.xMin ; x <= wall.xMax ; x++)
				{
					const Coordinates currentNodeCoordinates( x,y,z ) ;

					propagateNodeAtTileBorder
						(
							tile, currentNodeCoordinates, neighborTileDirection
						) ;
				}
	}
}



TEMPLATE_TILE_CALCULATOR
void TILE_CALCULATOR_CPU::
propagate( TileType & tile )
{
	auto nodeTypes   = tile.getNodeTypes() ;
	auto f           = tile.f() ;
	auto fPost       = tile.fPost() ;

	const Size tileSize(Edge, Edge, Edge) ;

	for (unsigned z=0 ; z < Edge ; z++)
		for (unsigned y=0 ; y < Edge ; y++)
			for (unsigned x=0 ; x < Edge ; x++)
			{
				auto currentNode = tile.getNode (x,y,z) ;
				NodeType nodeType = currentNode.nodeType() ;

				if ( not nodeType.isSolid() )
				{
					for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
					{
						auto const c = LatticeArrangement::c[q] ;
						Coordinates currentNodeCoordinates = Coordinates(x,y,z) ;

						// upwind
						Coordinates neighborNodeCoordinates = currentNodeCoordinates - c ;

						if ( tileSize.areCoordinatesInLimits( neighborNodeCoordinates ) )
						{
							unsigned nx = neighborNodeCoordinates.getX() ;
							unsigned ny = neighborNodeCoordinates.getY() ;
							unsigned nz = neighborNodeCoordinates.getZ() ;

							auto neighborNode = tile.getNode (nx,ny,nz) ;

							if ( not neighborNode.nodeType().isSolid() )
							{
								currentNode.f (q) = neighborNode.fPost (q) ;
							}
						}
					}
				}
			}

		// Walls
		propagateExternalNodes( tile, Cuboid(0,Edge-1, Edge-1,Edge-1, 0,Edge-1), Direction::NORTH ) ;
		propagateExternalNodes( tile, Cuboid(0,Edge-1, 0,0, 0,Edge-1), Direction::SOUTH ) ;
		propagateExternalNodes( tile, Cuboid(0,Edge-1, 0,Edge-1, Edge-1,Edge-1), Direction::TOP ) ;
		propagateExternalNodes( tile, Cuboid(0,Edge-1, 0,Edge-1, 0,0), Direction::BOTTOM ) ;
		propagateExternalNodes( tile, Cuboid(Edge-1,Edge-1, 0,Edge-1, 0,Edge-1), Direction::EAST ) ;
		propagateExternalNodes( tile, Cuboid(0,0, 0,Edge-1, 0,Edge-1), Direction::WEST ) ;
		
		// Edges
		propagateExternalNodes( tile, Cuboid(0,Edge-1, Edge-1,Edge-1, Edge-1,Edge-1), NT ) ;
		propagateExternalNodes( tile, Cuboid(0,Edge-1, Edge-1,Edge-1, 0,0), NB ) ;
		propagateExternalNodes( tile, Cuboid(Edge-1,Edge-1, Edge-1,Edge-1, 0,Edge-1), NE ) ;
		propagateExternalNodes( tile, Cuboid(0,0, Edge-1,Edge-1, 0,Edge-1), NW ) ;

		propagateExternalNodes( tile, Cuboid(0,Edge-1, 0,0, Edge-1,Edge-1), ST ) ;
		propagateExternalNodes( tile, Cuboid(0,Edge-1, 0,0, 0,0), SB ) ;
		propagateExternalNodes( tile, Cuboid(Edge-1,Edge-1, 0,0, 0,Edge-1), SE ) ;
		propagateExternalNodes( tile, Cuboid(0,0, 0,0, 0,Edge-1), SW ) ;

		propagateExternalNodes( tile, Cuboid(Edge-1,Edge-1, 0,Edge-1, Edge-1,Edge-1), ET ) ;
		propagateExternalNodes( tile, Cuboid(Edge-1,Edge-1, 0,Edge-1, 0,0), EB ) ;
		propagateExternalNodes( tile, Cuboid(0,0, 0,Edge-1, Edge-1,Edge-1), WT ) ;
		propagateExternalNodes( tile, Cuboid(0,0, 0,Edge-1, 0,0), WB ) ;
}



#undef TEMPLATE_TILE_CALCULATOR
#undef TILE_CALCULATOR_CPU



}



#endif
