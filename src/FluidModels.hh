#ifndef FLUID_MODELS_HH
#define FLUID_MODELS_HH



#include "Exceptions.hpp"



namespace microflow
{



template< class LatticeArrangement, class DataType >
const std::string FluidModelIncompressible< LatticeArrangement, DataType >::
getName()
{
	return "FluidModelIncompressible" ;
}



template< class LatticeArrangement, class DataType >
HD
DataType FluidModelIncompressible< LatticeArrangement, DataType >::
computeFeq
( 
 DataType rho, 
 DataType u [LatticeArrangement::getD()],
 Direction::DirectionIndex directionIndex
)
{
	Direction c = LatticeArrangement::getC( directionIndex ) ;
	constexpr unsigned D = LatticeArrangement::getD() ;
	DataType cu = 0 ;
	DataType u2 = 0 ;

	if (3 == D)
	{
		cu = c.getX() * u[0] + c.getY() * u[1] + c.getZ() * u[2] ;
		u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2] ;
	}
	else
	{
		THROW( "Unsupported value of D" ) ;
	}
	
	DataType w = LatticeArrangement::getW( directionIndex ) ;
	constexpr DataType csq = LatticeArrangement::csq ;
	DataType r = w * ( rho + 1.0 /csq * cu + 1.0 / (2.0 * csq * csq) * cu * cu - 1.0/(2.0 * csq) * u2);

	return r ;
}



template< class LatticeArrangement, class DataType >
const std::string FluidModelQuasicompressible< LatticeArrangement, DataType >::
getName()
{
	return "FluidModelQuasicompressible" ;
}



template< class LatticeArrangement, class DataType >
HD
DataType FluidModelQuasicompressible< LatticeArrangement, DataType >::
computeFeq
( 
 DataType rho, 
 DataType u [LatticeArrangement::getD()],
 Direction::DirectionIndex directionIndex
)
{
	Direction c = LatticeArrangement::getC( directionIndex ) ;
	constexpr unsigned D = LatticeArrangement::getD() ;
	DataType cu = 0 ;
	DataType u2 = 0 ;

	if (3 == D)
	{
		cu = c.getX() * u[0] + c.getY() * u[1] + c.getZ() * u[2] ;
		u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2] ;
	}
	else
	{
		THROW( "Unsupported value of D" ) ;
	}

	DataType w = LatticeArrangement::getW( directionIndex ) ;
	constexpr DataType csq = LatticeArrangement::csq ;
	DataType r = w * rho * ( 1.0 + 1.0 /csq * cu + 1.0 / (2.0 * csq * csq) * cu * cu - 1.0/(2.0 * csq) * u2);

	return r ;
}



}


#endif
