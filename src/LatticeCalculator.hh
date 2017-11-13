#ifndef LATTICE_CALCULATOR_HH
#define LATTICE_CALCULATOR_HH



#include "TileCalculator.hpp"



namespace microflow
{



#define TEMPLATE_LATTICE_CALCULATOR                           \
template<                                                     \
					template<class LatticeArrangement, class DataType>  \
														class FluidModel,                 \
														class CollisionModel,             \
														class LatticeArrangement,         \
														class DataType,                   \
														TileDataArrangement DataArrangement >



#define LATTICE_CALCULATOR_CPU   \
LatticeCalculator <FluidModel, CollisionModel, LatticeArrangement, DataType, StorageOnCPU, \
									 DataArrangement>



template< class DataType >
std::ostream & 
operator<<(std::ostream& out, 
					 const ComputationError<DataType> & computationError)
{
	out << "error = " << computationError.error
			<< ", maxVelocityLB = " << computationError.maxVelocityLB
			<< " at node " << computationError.maxVelocityNodeCoordinates ;

	return out ;
}



TEMPLATE_LATTICE_CALCULATOR
LATTICE_CALCULATOR_CPU::
LatticeCalculator (DataType rho0LB, 
									 DataType u0LB[ LatticeArrangement::getD()],
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



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_CPU::
initializeAtEquilibrium( TiledLatticeType & tiledLattice )
{
	processTiles< InitializatorAtEquilibrium >( tiledLattice ) ;

	tiledLattice.setValidCopyIDToF() ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_CPU::
collide( TiledLatticeType & tiledLattice )
{
	processTiles< Collider >( tiledLattice ) ;

	tiledLattice.setValidCopyIDToFPost() ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_CPU::
propagate( TiledLatticeType & tiledLattice )
{
	processTiles< Propagator >( tiledLattice ) ;

	tiledLattice.setValidCopyIDToF() ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_CPU::
processBoundary( TiledLatticeType & tiledLattice )
{
	processTiles< BoundaryProcessor >( tiledLattice ) ;

	tiledLattice.setValidCopyIDToF() ;
}



TEMPLATE_LATTICE_CALCULATOR
//TODO: ugly hack to avoid uT0 update after checkpoint load
template< bool updateVelocityT0 > 
ComputationError<DataType> LATTICE_CALCULATOR_CPU::
computeError( TiledLatticeType & tiledLattice )
{
	if ( 3 != LatticeArrangement::getD() )
	{
		THROW("UNIMPLEMENTED") ;
	}

	ComputationError<DataType> error ;
	error.error = 0 ;
	error.maxVelocityLB = 0 ;

	DataType e1(0), e2(0) ;


	for (auto t = tiledLattice.getBeginOfTiles() ;
						t < tiledLattice.getEndOfTiles() ;
						t++)
	{
		auto tile = tiledLattice.getTile(t) ;

		const unsigned Edge = tile.getNNodesPerEdge() ;
		for (unsigned z=0 ; z < Edge ; z++)
			for (unsigned y=0 ; y < Edge ; y++)
				for (unsigned x=0 ; x < Edge ; x++)
				{
					auto node = tile.getNode( x,y,z ) ;

					if ( node.nodeType().isSolid()  || 
							 node.nodeType() == NodeBaseType::BOUNCE_BACK_2 ) //FIXME: remove when rho computed
					 {
						 continue ;
					 }

					#define u   node.u
					#define uT0 node.uT0
					
					e1 += sqrt( 
											(u(X) - uT0(X)) * (u(X) - uT0(X)) +
											(u(Y) - uT0(Y)) * (u(Y) - uT0(Y)) +
											(u(Z) - uT0(Z)) * (u(Z) - uT0(Z)) 
										) ;
					e2 += sqrt(	u(X) * u(X) + u(Y) * u(Y) + u(Z) * u(Z) ) ;
					
					if (updateVelocityT0)
					{
						uT0(X) = u(X) ;
						uT0(Y) = u(Y) ;
						uT0(Z) = u(Z) ;
					}

					bool maxVelocityFound = false ;

					if ( fabs(u(X)) > fabs(error.maxVelocityLB) )
					{
						error.maxVelocityLB = u(X) ;
						maxVelocityFound = true ;
					}
					if ( fabs(u(Y)) > fabs(error.maxVelocityLB) )
					{
						error.maxVelocityLB = u(Y) ;
						maxVelocityFound = true ;
					}
					if ( fabs(u(Z)) > fabs(error.maxVelocityLB) )
					{
						error.maxVelocityLB = u(Z) ;
						maxVelocityFound = true ;
					}

					if ( maxVelocityFound )
					{
						//TODO: tiledLattice.getCornerPosition() ?
						auto tileLayout = tiledLattice.getTileLayout() ;
						auto tileIndex = tile.getCurrentTileIndex() ;
						Coordinates tileCorner = tileLayout.getTile(tileIndex).getCornerPosition() ;

						error.maxVelocityNodeCoordinates = tileCorner + Coordinates(x,y,z) ;
					}

					#undef u
					#undef uT0
				}
	}

	error.error = e1 / e2 ;

	return error ;
}



TEMPLATE_LATTICE_CALCULATOR
void LATTICE_CALCULATOR_CPU::
computeRhoForBB2Nodes (TiledLatticeType & tiledLattice)
{
	processTiles <RhoBB2Calculator> (tiledLattice) ;
}



TEMPLATE_LATTICE_CALCULATOR
typename LATTICE_CALCULATOR_CPU::TileCalculatorType
LATTICE_CALCULATOR_CPU::
getTileCalculator()
{
	return TileCalculatorType (this->rho0LB_, this->u0LB_, this->tau_,
															this->defaultExternalEdgePressureNode_,
															this->defaultExternalCornerPressureNode_
														 ) ;
}



TEMPLATE_LATTICE_CALCULATOR
template< class Operator >
void LATTICE_CALCULATOR_CPU::
processTiles( TiledLatticeType & tiledLattice )
{
	auto tileCalculator = getTileCalculator() ;

	for (auto t = tiledLattice.getBeginOfTiles() ;
						t < tiledLattice.getEndOfTiles() ;
						t++)
	{
		auto tile = tiledLattice.getTile(t) ;

		Operator::apply( tileCalculator, tile ) ;
	}
}



#undef TEMPLATE_LATTICE_CALCULATOR
#undef LATTICE_CALCULATOR_CPU



}



#endif
