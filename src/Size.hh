#ifndef SIZE_HH
#define SIZE_HH



#include <cstddef>
#include <algorithm>



namespace microflow
{



INLINE 
HD
Size::
Size( size_t width, size_t height, size_t depth )
: Coordinates( width, height, depth )
{
}



INLINE
HD
Size::
Size()
: Coordinates()
{
}



INLINE
HD
size_t Size::
computeVolume() const
{
	return getWidth() * getHeight() * getDepth() ;
}



INLINE
HD
bool Size::
areCoordinatesInLimits(size_t x, size_t y, size_t z) const
{
	return ( (x < getWidth()) && (y < getHeight()) && (z < getDepth()) ) ;
}



INLINE
HD
bool Size::
areCoordinatesInLimits(Coordinates coordinates) const
{
	return areCoordinatesInLimits( coordinates.getX(), 
																 coordinates.getY(), 
																 coordinates.getZ() ) ;
}



INLINE
HD
size_t Size::
getWidth() const
{
	return getX() ;
}



INLINE
HD
size_t Size::
getHeight() const
{
	return getY() ;
}



INLINE
HD
size_t Size::
getDepth() const
{
	return getZ() ;
}



inline
Size max( const Size & s1, const Size & s2 )
{
	return Size(
								std::max( s1.getWidth (), s2.getWidth () ),
								std::max( s1.getHeight(), s2.getHeight() ),
								std::max( s1.getDepth (), s2.getDepth () )
						 ) ;
}



INLINE
HD
size_t 
linearizeXYZ(const Coordinates & coordinates, const Size & size )
{
	//FIXME: probably used somewere for program logic, tests fail with -DNDEBUG
	ASSERT( size.areCoordinatesInLimits( coordinates ) ) ;

	size_t x = coordinates.getX() ;
	size_t y = coordinates.getY() ;
	size_t z = coordinates.getZ() ;

	size_t linearIndex = x + 
											 y * size.getWidth() + 
											 z * size.getWidth() * size.getHeight() ;

	return linearIndex ;	
}



}



#endif
