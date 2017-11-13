#ifndef TILE_CALCULATOR_HPP
#define TILE_CALCULATOR_HPP



#include "Tile.hpp"
#include "Storage.hpp"
#include "Calculator.hpp"
#include "NodeCalculator.hpp"
#include "Cuboid.hpp"


namespace microflow
{



template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement, 
														class DataType,
														unsigned Edge,
					template<class T> class Storage,
														TileDataArrangement DataArrangement 
				>
class TileCalculator ;



template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement, 
														class DataType,
														unsigned Edge,
														TileDataArrangement DataArrangement >
class TileCalculator< 
											FluidModel, CollisionModel, LatticeArrangement, 
											DataType, Edge, StorageOnCPU, DataArrangement
										>
: protected Calculator< FluidModel, LatticeArrangement, DataType, StorageOnCPU >
{
	public:
		TileCalculator (DataType rho0LB, 
										DataType u0LB[LatticeArrangement::getD()],
										DataType tau,
										NodeType defaultExternalEdgePressureNode,
										NodeType defaultExternalCornerPressureNode 
									 ) ;
		typedef Tile<LatticeArrangement, DataType, Edge, StorageOnCPU, DataArrangement>  TileType ;
		typedef Calculator<FluidModel, LatticeArrangement, DataType, StorageOnCPU>  CalculatorType ;
		
		void initializeAtEquilibrium( TileType & tile ) ;
		void collide                ( TileType & tile ) ;
		void propagate              ( TileType & tile ) ;
		void processBoundary        ( TileType & tile ) ;

		void computeRhoForBB2Nodes (TileType & tile) ;


	protected:
		void propagateExternalNodes( TileType & tile, 
																 Cuboid nodes, 
																 Direction neighborTileDirection ) ;

		template< class Operator >
		void processNodes( TileType & tile ) ;

		const NodeType defaultExternalEdgePressureNode_ ;
		const NodeType defaultExternalCornerPressureNode_ ; 
} ;



template< class TileType >
HD
void propagateNodeAtTileBorder( TileType & tile,
																Coordinates currentNodeCoordinates,
																Direction neighborTileDirection ) ;



}



#include "TileCalculator.hh"



#endif
