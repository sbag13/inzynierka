#ifndef FLUID_MODELS_HPP
#define FLUID_MODELS_HPP



#include "Direction.hpp"
#include "cudaPrefix.hpp"



namespace microflow
{



class FluidModel
{
	public:

		static constexpr bool isIncompressible = false ;
		static constexpr bool isCompressible   = false ;
} ;



template< class LatticeArrangement, class DataType >
class FluidModelIncompressible : public FluidModel
{
	public:

		static constexpr bool isIncompressible = true ;
		static const std::string getName() ;

		HD static DataType 
		computeFeq( 
								DataType rho, 
								DataType u [LatticeArrangement::getD()],
								Direction::DirectionIndex directionIndex
							) ;

} ;



template< class LatticeArrangement, class DataType >
class FluidModelQuasicompressible : public FluidModel
{
	public:

		static constexpr bool isCompressible = true ;
		static const std::string getName() ;

		HD static DataType 
		computeFeq( 
								DataType rho, 
								DataType u [LatticeArrangement::getD()],
								Direction::DirectionIndex directionIndex
							) ;

} ;



}



#include "FluidModels.hh"



#endif
