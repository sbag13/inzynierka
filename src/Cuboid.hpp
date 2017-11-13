#ifndef CUBOID_HPP
#define CUBOID_HPP



namespace microflow
{


class Cuboid
{
	public:

		constexpr
		Cuboid( unsigned xMin_, unsigned xMax_,
						unsigned yMin_, unsigned yMax_,
						unsigned zMin_, unsigned zMax_ ) ;

		unsigned xMin ;
		unsigned xMax ;
		unsigned yMin ;
		unsigned yMax ;
		unsigned zMin ;
		unsigned zMax ;
} ;



}



#include "Cuboid.hh"



#endif
