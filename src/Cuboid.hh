#ifndef CUBOID_HH
#define CUBOID_HH



namespace microflow
{



inline
constexpr
Cuboid::
Cuboid( unsigned xMin_, unsigned xMax_,
				unsigned yMin_, unsigned yMax_,
				unsigned zMin_, unsigned zMax_ )
: xMin(xMin_), xMax(xMax_), yMin(yMin_), yMax(yMax_), zMin(zMin_), zMax(zMax_)
{
}



}



#endif
