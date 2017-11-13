#ifndef NODE_CALCULATOR_HPP
#define NODE_CALCULATOR_HPP



#include "LatticeArrangement.hpp"
#include "Calculator.hpp"
#include "FluidModels.hpp"
#include "Optimization.hpp"
#include "NodeType.hpp"



namespace microflow
{



// Used to specify, where save result of collision.
enum class WhereSaveF
{
	F,
	F_POST
} ;



template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class LatticeArrangement, 
														class DataType,
					template<class T> class Storage >
class NodeCalculatorBase :
protected Calculator< FluidModel, LatticeArrangement, DataType, Storage >
{
	public:

		HD NodeCalculatorBase(DataType rho0LB, 
									 DataType u0LB[LatticeArrangement::getD()],
									 DataType tau 
								#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
									 , DataType invRho0LB
									 , DataType invTau 
								#endif
									 ) ;

		template< class Node >
		HD void initializeAtEquilibrium( Node & node ) ;

	protected:

		typedef Calculator< FluidModel, LatticeArrangement, DataType, Storage > CalculatorType ;

		template <class Node, WhereSaveF whereSaveF>
		HD void collideBGK (Node & node) ;

		template <class Node, WhereSaveF whereSaveF>
		HD void collideBounceBack2 (Node & node) ;

		template <class Node , class Optimization>
		HD void processBoundaryFluid( Node & node ) ;

		template <class Node>
		HD void processBoundaryFluid_NoOptimizations (Node & node) ;
		template <class Node>
		HD void processBoundaryFluid_UnsafeOptimizations (Node & node) ;

		template< class Node >
		void processBoundaryBounceBack2( Node & node ) ;
	
#ifdef __CUDACC__
		// GPU version needs information about grid configuration.
		template <class ThreadMapper, class Node>
		DEVICE void processBoundaryBounceBack2 (Node & node) ;
#endif
} ;




template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement, 
														class DataType,
					template<class T> class Storage >
class NodeCalculator ;



template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class DataType,
					template<class T> class Storage >
class NodeCalculator< FluidModel, CollisionModel, D3Q19, DataType, Storage > :
public NodeCalculatorBase< FluidModel, D3Q19, DataType, Storage >
{
	public:

		HD NodeCalculator (DataType rho0LB, 
											 DataType u0LB[D3Q19::getD()],
											 DataType tau, 
										#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
											 DataType invRho0LB ,
											 DataType invTau ,
										#endif
											 NodeType defaultExternalEdgePressureNode,
											 NodeType defaultExternalCornerPressureNode
											) ;

		template <class Node, WhereSaveF whereSaveF = WhereSaveF::F_POST >
		HD void collide( Node & node ) ;

		template< 
						#ifdef __CUDA_ARCH__
							class ThreadMapper,
						#endif
							class Node ,
						#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
							class Optimization = UnsafeOptimizations
						#else
							class Optimization = NoOptimizations
						#endif
						>
		HD void processBoundary( Node & node ) ;


	protected:

		typedef NodeCalculatorBase< FluidModel, D3Q19, DataType, Storage > 
							BaseCalculator ;

		template <class Node, WhereSaveF whereSaveF>
		HD void collideMRT( Node & node ) ;

		template
		< 
			template< class, class > class FluidModelType, 
			int Dummy
		>
		class MEQCalculator ;
		// Dummy avoids complete specialization of a nested class
		// http://stackoverflow.com/questions/3052579/explicit-specialization-in-non-namespace-scope

		template< int Dummy >
		class MEQCalculator< FluidModelIncompressible, Dummy>
		{
			public:
				static HD DataType computeMEQ
				( 
					DataType rho, 
					DataType u, DataType v, DataType w,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
					DataType invRho0LB,
#else
					DataType rho0LB, 
#endif
					int k 
				) ;
		} ;
		
		template< int Dummy >
		class MEQCalculator< FluidModelQuasicompressible, Dummy>
		{
			public:
				static HD DataType computeMEQ
				( 
					DataType rho, 
					DataType u, DataType v, DataType w,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
					DataType invRho0LB,
#else
					DataType rho0LB, 
#endif
					int k 
				) ;
		} ;


	private:

		template< class Node >
		HD void processBoundaryVelocity0ExternalEdge( Node & node ) ;

		template< class Node >
		HD void processBoundaryVelocity0InternalEdge( Node & node ) ;

		template< class Node >
		HD void processBoundaryVelocity0ExternalCorner( Node & node ) ;

		template< class Node >
		HD void processBoundaryVelocity0CornerSurf( Node & node ) ;

		template< class Node >
		HD void bounceBackLackingDirections( Node & node ) ;

		const NodeType defaultExternalEdgePressureNode_ ;
		const NodeType defaultExternalCornerPressureNode_ ;
} ;



}



#include "NodeCalculator.hh"



#endif
