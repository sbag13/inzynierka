#ifndef COORDINATES_HH
#define COORDINATES_HH



namespace microflow
{



template <class T>
INLINE HD
UniversalCoordinates<T>::
UniversalCoordinates( T x, T y, T z )
: x_(x), y_(y), z_(z)
{}



template <class T>
INLINE HD
UniversalCoordinates<T>::
UniversalCoordinates( const Direction & direction )
: x_( direction.getX() ), 
  y_( direction.getY() ), 
	z_( direction.getZ() )
{}



template <class T>
INLINE HD
UniversalCoordinates<T>::
UniversalCoordinates()
: x_(0), y_(0), z_(0)
{}



template <class T>
INLINE HD
T UniversalCoordinates<T>::
getX() const
{
	return x_ ;
}



template <class T>
INLINE HD
T UniversalCoordinates<T>::
getY() const
{
	return y_ ;
}



template <class T>
INLINE HD
T UniversalCoordinates<T>::
getZ() const
{
	return z_ ;
}



template <class T>
INLINE HD
void UniversalCoordinates<T>::
setX( T x )
{
	x_ = x ;
}



template <class T>
INLINE HD
void UniversalCoordinates<T>::
setY( T y )
{
	y_ = y ;
}



template <class T>
INLINE HD
void UniversalCoordinates<T>::
setZ( T z )
{
	z_ = z ;
}



template <class T>
INLINE HD
UniversalCoordinates<T> UniversalCoordinates<T>::
operator+( const UniversalCoordinates & coordinates ) const
{
	return UniversalCoordinates<T>( getX() + coordinates.getX(),
											getY() + coordinates.getY(),
											getZ() + coordinates.getZ() ) ;
}



template <class T>
INLINE HD
UniversalCoordinates<T> UniversalCoordinates<T>::
operator-( const UniversalCoordinates & coordinates ) const
{
	return UniversalCoordinates<T>( getX() - coordinates.getX(),
											getY() - coordinates.getY(),
											getZ() - coordinates.getZ() ) ;
}



template <class T>
INLINE HD
UniversalCoordinates<T> UniversalCoordinates<T>::
operator-( const Direction & direction ) const
{
	return operator-( UniversalCoordinates<T>(direction) ) ;
}



template <class T>
INLINE HD
bool UniversalCoordinates<T>::
operator!=( const UniversalCoordinates<T> & coordinates ) const
{
	if (coordinates.getX() != getX() ) return true ;
	if (coordinates.getY() != getY() ) return true ;
	if (coordinates.getZ() != getZ() ) return true ;

	return false ;
}



template <class T>
INLINE HD
bool UniversalCoordinates<T>::
operator==( const UniversalCoordinates<T> & coordinates ) const
{
	return (not operator!=( coordinates )) ;
}



template <class T>
INLINE
HD T 
dotProduct (const UniversalCoordinates<T> & c1, const UniversalCoordinates<T> & c2)
{
	return c1.getX() * c2.getX() + c1.getY() * c2.getY() + c1.getZ() * c2.getZ() ;
}



template <class T>
inline
std::ostream & 
operator<<(std::ostream& out, const UniversalCoordinates<T> & coordinates)
{
	out << "(" ;
	out << "x=" << coordinates.getX() ;
	out << ", y=" << coordinates.getY() ;
	out << ", z=" << coordinates.getZ() ;
	out << ")" ;
	return out ;
}



template <class T>
inline
std::istream & 
operator>> (std::istream& input, UniversalCoordinates<T> & coordinates)
{
	T tmp ;

	input >> tmp ; coordinates.setX (tmp) ;
	input >> tmp ; coordinates.setY (tmp) ;
	input >> tmp ; coordinates.setZ (tmp) ;

  return input ;
}



}



#endif
