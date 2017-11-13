#ifndef LBM_OPERATOR_CHOOSER_HPP
#define LBM_OPERATOR_CHOOSER_HPP



#include "gpuTools.hpp"



namespace microflow
{



class InitializatorAtEquilibrium
{
	public:
		template< class Calculator, class Element >
		HD
		static void apply( Calculator & calculator, 
											 Element    & element ) ;
} ;



class Collider
{
	public:
		template< class Calculator, class Element >
		HD
		static void apply( Calculator & calculator, 
											 Element    & element ) ;
} ;



class Propagator
{
	public:
		template< class Calculator, class Element >
		HD
		static void apply( Calculator & calculator, 
											 Element    & element ) ;
} ;



class BoundaryProcessor
{
	public:
		// Only for CPU version, GPU version needs information about grid configuration.
		template< class Calculator, class Element >
		static void apply( Calculator & calculator, 
											 Element    & element ) ;
} ;



class RhoBB2Calculator
{
	public:
		template <class Calculator, class Element>
		static void apply (Calculator & calculator, 
											 Element    & element) ;
} ;



}



#include "LBMOperatorChooser.hh"



#endif
