#ifndef COORDINATES_HPP
#define COORDINATES_HPP



#include <iostream>

#include "Direction.hpp"



namespace microflow
{



template <class T = size_t>
class UniversalCoordinates
{
	public:

		HD UniversalCoordinates (T x, T y, T z = 1) ;
		HD UniversalCoordinates (const Direction & direction) ;
		HD UniversalCoordinates() ;

		HD T getX() const ;
		HD T getY() const ;
		HD T getZ() const ;

		HD void setX (T x) ;
		HD void setY (T y) ;
		HD void setZ (T z) ;

		HD UniversalCoordinates<T> operator+ (const UniversalCoordinates<T> & coordinates) const ;
		HD UniversalCoordinates<T> operator- (const UniversalCoordinates<T> & coordinates) const ;
		HD UniversalCoordinates<T> operator- (const Direction & direction) const ;
		
		HD bool operator!= (const UniversalCoordinates<T> & coordinates) const ;
		HD bool operator== (const UniversalCoordinates<T> & coordinates) const ;


	private:

		T x_ ;
		T y_ ;
		T z_ ;
} ;



typedef UniversalCoordinates <size_t> Coordinates ;



template <class T>
INLINE
HD T dotProduct (const UniversalCoordinates<T> & c1, const UniversalCoordinates<T> & c2) ;



template <class T>
std::ostream & operator<< (std::ostream & out, const UniversalCoordinates<T> & coordinates) ;

template <class T>
std::istream & operator>> (std::istream & input, UniversalCoordinates<T> & coordinates) ;



}



#include "Coordinates.hh"



#endif
