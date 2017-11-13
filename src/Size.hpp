#ifndef SIZE_HPP
#define SIZE_HPP



#include "Coordinates.hpp"



namespace microflow
{

class Size : public Coordinates
{
	public:
		HD Size(size_t width, size_t height, size_t depth = 1 ) ;
		HD Size() ;

		HD size_t getWidth() const ;
		HD size_t getHeight() const ;
		HD size_t getDepth() const ;

		HD size_t computeVolume() const ;

		HD bool areCoordinatesInLimits(size_t x, size_t y, size_t z) const;
		HD bool areCoordinatesInLimits(Coordinates coordinates) const;
} ;



Size max( const Size & s1, const Size & s2 ) ;

HD size_t linearizeXYZ(const Coordinates & coordinates, const Size & size ) ;


}



#include "Size.hh"



#endif
