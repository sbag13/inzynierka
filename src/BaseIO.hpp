#ifndef BASE_IO_HPP
#define BASE_IO_HPP



#include <sstream>

#include "Direction.hpp"



namespace microflow
{



template <class LatticeArrangement>
inline
std::string
buildFArrayName (const std::string & prefix,
								 const Direction::DirectionIndex & directionIndex)
{
	std::stringstream ss ;

	auto d = LatticeArrangement::getC (directionIndex) ;
	ss << prefix << directionIndex << "(" << toString(d) << ")" ;
	return ss.str() ;
}



}



#endif
