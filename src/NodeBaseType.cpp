#include "NodeBaseType.hpp"

#include <map>
#include <stdexcept>



namespace microflow
{



static std::map< NodeBaseType, std::string > nodeBaseTypeNames = 
{
   {NodeBaseType::SOLID         , "solid"            } ,
   {NodeBaseType::FLUID         , "fluid"            } ,
   {NodeBaseType::BOUNCE_BACK_2 , "bounce_back_2"    } ,
   {NodeBaseType::VELOCITY      , "velocity"         } ,
   {NodeBaseType::VELOCITY_0    , "velocity_0"       } ,
	 {NodeBaseType::PRESSURE      , "pressure"         } ,
	 {NodeBaseType::MARKER        , "WARNING - marker" } ,
	 {NodeBaseType::SIZE          , "WARNING - size"   } 
};



const std::string & toString( NodeBaseType nodeBaseType )
{
	return nodeBaseTypeNames.at( nodeBaseType ) ;
}



template<>
NodeBaseType fromString<NodeBaseType>( const std::string & name )
{
	for (auto it  = nodeBaseTypeNames.cbegin() ; 
						it != nodeBaseTypeNames.end() ; 
						++it)
	{
		if ( it->second == name ) 
		{
			return it->first ;
		}
	}
	throw std::out_of_range("Name not found in nodeBaseTypeNames") ;
}



std::ostream & operator<<(std::ostream & os, const NodeBaseType & nodeBaseType)
{
	os << toString( nodeBaseType ) ;
	return os ;
}



}
