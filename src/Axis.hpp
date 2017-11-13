#ifndef AXIS_HPP
#define AXIS_HPP



namespace microflow
{



enum class Axis
{
	X = 0,
	Y = 1,
	Z = 2
} ;



static constexpr unsigned X = 0 ;
static constexpr unsigned Y = 1 ;
static constexpr unsigned Z = 2 ;



inline
enum Axis toAxis(unsigned i)
{
	switch (i)
	{
		case 0: return Axis::X ;
		case 1: return Axis::Y ;
		case 2: return Axis::Z ;
	}

	throw "Invalid axis number" ;
}



}



#endif
