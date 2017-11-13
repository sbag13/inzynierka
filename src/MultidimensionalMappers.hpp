#ifndef MULTIDIMENSIONAL_MAPPERS_HPP
#define MULTIDIMENSIONAL_MAPPERS_HPP

/*
	Algorithms used for calculation of linear index from multidimensional coordinates.

	Needed mainly for defining of data arrangement, i.e. order of f_i values inside tile.
*/


#include "cudaPrefix.hpp"


namespace microflow
{



/*
	The simplest arrangement, data stored in row order.
*/
class XYZ
{
	public:

		HD static	constexpr
		unsigned linearize (unsigned x, unsigned y, unsigned z,
												unsigned width, unsigned height, unsigned depth) ;
} ;



/*
	Similar to XYZ, but with X and Y axes swapped.
*/
class YXZ
{
	public:

		HD static	constexpr
		unsigned linearize (unsigned x, unsigned y, unsigned z,
												unsigned width, unsigned height, unsigned depth) ;
} ;



/*
	WARNING: Works only for 4x4x4 tile.


  9  10  11  12
  6   7   8  13
  3   4   5  14
  0   1   2  15
*/
class ZigzagNE
{
	public:

		HD static	constexpr
		unsigned linearize (unsigned x, unsigned y, unsigned z,
												unsigned width, unsigned height, unsigned depth) ;
} ;



}



#include "MultidimensionalMappers.hh"



#endif
