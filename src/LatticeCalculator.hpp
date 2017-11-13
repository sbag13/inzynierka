#ifndef LATTICE_CALCULATOR_HPP
#define LATTICE_CALCULATOR_HPP



#include "Tile.hpp"
#include "Storage.hpp"
#include "TiledLattice.hpp"
#include "Calculator.hpp"
#include "TileCalculator.hpp"
#include "LBMOperatorChooser.hpp"
#include "DataFlowDirection.hpp"



namespace microflow
{



//TODO: LatticeCalculatorBase class with duplicated code. Probably CRTP is needed,
//			because processTiles() method must be different for CPU and GPU versions.

//TODO: Maybe LatticeCalculator template should have TiledLattice as parameter?
//			This could allow to "pack" 4 last parameters into one. 

template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement, 
														class DataType,
					template<class T> class Storage,
					TileDataArrangement DataArrangement>
class LatticeCalculator ;



//TODO: make it internal type of LatticeCalculator ?
template< class DataType >
class ComputationError
{
	public:

		DataType error ;
		DataType maxVelocityLB ;
		Coordinates maxVelocityNodeCoordinates ;
} ;



template< class DataType >
std::ostream& operator<<(std::ostream& out, 
												 const ComputationError<DataType> & computationError) ;



template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement, 
														class DataType,
														TileDataArrangement DataArrangement
				>
class LatticeCalculator< FluidModel, CollisionModel, LatticeArrangement, DataType, StorageOnCPU,
												 DataArrangement >
: protected Calculator< FluidModel, LatticeArrangement, DataType, StorageOnCPU >
{
	public:

		//FIXME: Reconsider passing full Settings() object instead of separate
		//			 defaultExternalEdgePressureNode and defaultExternalCornerPressureNode,
		//			 because for separate parameters there is no detection of wrong
		//			 values resulting from displacement (wrong positions).
		LatticeCalculator (DataType rho0LB, 
											 DataType u0LB[LatticeArrangement::getD()],
											 DataType tau,
											 NodeType defaultExternalEdgePressureNode,
											 NodeType defaultExternalCornerPressureNode 
										  ) ;

		typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement>
						TiledLatticeType ;

		void initializeAtEquilibrium( TiledLatticeType & tiledLattice ) ;
		void collide                ( TiledLatticeType & tiledLattice ) ;
		void processBoundary        ( TiledLatticeType & tiledLattice ) ;
		void propagate              ( TiledLatticeType & tiledLattice ) ;

		//TODO: ugly hack to avoid uT0 update after checkpoint load
		template< bool updateVelocityT0 = true >
		ComputationError<DataType> 
		computeError( TiledLatticeType & tiledLattice ) ;

		void computeRhoForBB2Nodes (TiledLatticeType & tiledLattice) ;


	protected:

		typedef Calculator< FluidModel, LatticeArrangement, DataType, StorageOnCPU > CalculatorType ;

		typedef	TileCalculator< FluidModel, 
									CollisionModel,
									LatticeArrangement, 
									DataType, 
									TiledLatticeType::getNNodesPerTileEdge(), 
									StorageOnCPU,
									DataArrangement > 
							TileCalculatorType ;

		TileCalculatorType getTileCalculator() ;

		template< class Operator >
		void processTiles( TiledLatticeType & tiledLattice ) ;

		//FIXME: Reconsider Settings() object instead of separate nodes - more error
		//        prone while passing to lower level calculators.
		const NodeType defaultExternalEdgePressureNode_ ;
		const NodeType defaultExternalCornerPressureNode_ ; 
} ;



template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement, 
														class DataType,
														TileDataArrangement DataArrangement>
class LatticeCalculator< FluidModel, CollisionModel, LatticeArrangement, DataType, StorageOnGPU,
												 DataArrangement >
: protected  Calculator< FluidModel, LatticeArrangement, DataType, StorageOnGPU >
{
	public:

		LatticeCalculator (DataType rho0LB, 
										 	 DataType u0LB[LatticeArrangement::getD()],
											 DataType tau,
											 NodeType defaultExternalEdgePressureNode,
											 NodeType defaultExternalCornerPressureNode 
											) ;
		
		typedef TiledLattice <LatticeArrangement, DataType, StorageOnGPU, DataArrangement> 
						TiledLatticeType ;

		void initializeAtEquilibrium( TiledLatticeType & tiledLattice ) ;
		void collide                ( TiledLatticeType & tiledLattice ) ;
		void processBoundary        ( TiledLatticeType & tiledLattice ) ;
		void propagate              ( TiledLatticeType & tiledLattice ) ;
	
		//FIXME: Replace propagate() with the below method when optimisations are finished.
		void propagateOpt       (TiledLatticeType & tiledLattice) ;
		void collideOpt         (TiledLatticeType & tiledLattice) ;
		void processBoundaryOpt (TiledLatticeType & tiledLAttice) ;

		// All computation steps combined int single kernel.
		void initializeAtEquilibriumForGather (TiledLatticeType & tiledLattice) ;
		void gatherProcessBoundaryCollide (TiledLatticeType & tiledLattice,
																			 bool shouldComputeRhoU) ;

		// FIXME: Only for tests, remove when not needed.
		void swapFPostWithF (TiledLatticeType & tiledLattice) ;


	private:

		typedef Calculator< FluidModel, LatticeArrangement, DataType, StorageOnGPU > CalculatorType ;

		template< class Operator >
		void processTiles( TiledLatticeType & tiledLattice ) ;

		const NodeType defaultExternalEdgePressureNode_ ;
		const NodeType defaultExternalCornerPressureNode_ ; 

		template <DataFlowDirection, class ShouldSaveRhoU>
		void callKernelTileGatherProcessBoundaryCollide
			(TiledLatticeType & tiledLattice) ;
} ;



}



#include "LatticeCalculator.hh"
#include "LatticeCalculator.tcc"



#endif
