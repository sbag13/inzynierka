#ifndef CALCULATOR_HPP
#define CALCULATOR_HPP



#include "cudaPrefix.hpp"



namespace microflow
{



//FIXME: Leave only LatticeArrangement and DataType as template arguments ?
template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class LatticeArrangement, 
														class DataType,
					template<class T> class Storage >
class Calculator
{
	public:

		HD Calculator( DataType rho0LB, 
									 DataType u0LB[LatticeArrangement::getD()],
									 DataType tau
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
									 , DataType invRho0LB
									 , DataType invTau 
#endif
									 ) ;


	protected: //TODO: private and getters ?

		DataType rho0LB_ ;
		DataType u0LB_[ LatticeArrangement::getD() ] ;
		DataType tau_ ;
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
		DataType invTau_ ;
		DataType invRho0LB_ ;
#endif
} ;



}



#include "Calculator.hh"



#endif
