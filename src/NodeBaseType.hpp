#ifndef NODE_BASE_TYPE
#define NODE_BASE_TYPE



#include <string>

#include "gpuTools.hpp"



namespace microflow
{



enum class NodeBaseType 
{ 
	SOLID         = 0, 
	FLUID         = 1, 

	// Boundary types should have consecutive numbers for isBoundary() method.
	BOUNCE_BACK_2 = 2,
	VELOCITY      = 3,
	VELOCITY_0    = 4,
	PRESSURE      = 5,

	MARKER           ,

	SIZE
} ;



inline
HD bool isSolid( NodeBaseType const & type )
{
	return NodeBaseType::SOLID == type ;
}



inline
HD bool isFluid( NodeBaseType const & type )
{
	return NodeBaseType::FLUID == type ;
}



inline
HD bool isBoundary( NodeBaseType const & type )
{
	return (NodeBaseType::FLUID < type && NodeBaseType::MARKER > type) ;
}



const std::string & toString( NodeBaseType nodeBaseType ) ;


template <class T>
T fromString (const std::string & name) ;

template<>
NodeBaseType fromString<NodeBaseType>( const std::string & name ) ;

std::ostream & operator<<(std::ostream & os, const NodeBaseType & nodeBaseType) ;



}



#endif
