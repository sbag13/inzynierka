#ifndef DIRECTION_HH
#define DIRECTION_HH



#include <cstddef>
#include <cassert>
#include <limits>



namespace microflow
{



INLINE
HD
Direction::
Direction( Direction::D direction )
: PackedDirectionVector<DIRECTION_TYPE>(direction)
{
}



INLINE
HD
Direction::
Direction()
: PackedDirectionVector<DIRECTION_TYPE>()
{
}



INLINE
HD
Direction::D Direction::
get() const
{
	return PackedDirectionVector<DIRECTION_TYPE>::getInternalStorage() ;
}



INLINE
HD
int Direction::
getX() const
{
	return PackedDirectionVector<DIRECTION_TYPE>::getCoordinate(0) ;
}



INLINE
HD
int Direction::
getY() const
{
	return PackedDirectionVector<DIRECTION_TYPE>::getCoordinate(1) ;
}



INLINE
HD
int Direction::
getZ() const
{
	return PackedDirectionVector<DIRECTION_TYPE>::getCoordinate(2) ;
}



INLINE
HD
void Direction::
setX( int value )
{
	PackedDirectionVector<DIRECTION_TYPE>::setCoordinate(0, value) ;
}



INLINE
HD
void Direction::
setY( int value )
{
	PackedDirectionVector<DIRECTION_TYPE>::setCoordinate(1, value) ;
}



INLINE
HD
void Direction::
setZ( int value )
{
	PackedDirectionVector<DIRECTION_TYPE>::setCoordinate(2, value) ;
}



INLINE
HD
Direction Direction::
computeInverse() const
{
	Direction d(*this) ;

	d.setX( - getX() ) ;
	d.setY( - getY() ) ;
	d.setZ( - getZ() ) ;

	return d ;
}



INLINE
Direction::DirectionIndex Direction::
getIndexD3Q27() const
{
	// Should result in sgmentation fault
	constexpr DirectionIndex NO_INDEX = (DirectionIndex)(0) - (DirectionIndex)(1) ;

	// Remember to modify after changing the order in Direction::D3Q27 array
	constexpr DirectionIndex indexFromDirection_[ 0b00111111 + 1 ] =
	{
		NO_INDEX,        0, NO_INDEX,        2,        1,    6 + 0, NO_INDEX,    6 + 4, 
		NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,        3,    6 + 1, NO_INDEX,    6 + 5, 
					 4,    6 + 2, NO_INDEX,    6 + 6,    6 + 8,   18 + 0, NO_INDEX,   18 + 2, 
		NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,   6 + 10,   18 + 1, NO_INDEX,   18 + 3, 
		NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, 
		NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, 
					 5,    6 + 3, NO_INDEX,    6 + 7,    6 + 9,   18 + 4, NO_INDEX,   18 + 6, 
		NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,   6 + 11,   18 + 5, NO_INDEX,   18 + 7 
	} ;

	ASSERT ( get() <= 0b00111111 ) ;

	return indexFromDirection_[ static_cast<unsigned>(get()) ] ;
}



INLINE
HD
Direction Direction::
operator+( const Direction & right ) const
{
	Direction result ;

	result.setX( getX() + right.getX() ) ;
	result.setY( getY() + right.getY() ) ;
	result.setZ( getZ() + right.getZ() ) ;

	return result ;
}



INLINE
HD
Direction Direction::
operator-( const Direction & right ) const
{
	Direction result ;

	result.setX( getX() - right.getX() ) ;
	result.setY( getY() - right.getY() ) ;
	result.setZ( getZ() - right.getZ() ) ;

	return result ;
}



INLINE
HD bool Direction::
operator==( const Direction & direction ) const
{
	return ( this->get() == direction.get() ) ;
}



INLINE
HD bool Direction::
operator!=( const Direction & direction ) const
{
	return not this->operator== (direction) ;
}



INLINE 
HD Direction
crossProduct (const Direction & d1, const Direction & d2)
{
	Direction result ;

	result.setX (d1.getY() * d2.getZ() - d1.getZ() * d2.getY()) ;
	result.setY (d1.getZ() * d2.getX() - d1.getX() * d2.getZ()) ;
	result.setZ (d1.getX() * d2.getY() - d1.getY() * d2.getX()) ;

	return result ;
}



INLINE 
HD int
dotProduct (const Direction & d1, const Direction & d2)
{
	return d1.getX() * d2.getX() + d1.getY() * d2.getY() + d1.getZ() * d2.getZ() ;
}



inline
std::ostream& operator<<(std::ostream& out, const Direction & direction) 
{
	out << "{" ;

	out << " " << toString( direction ) << " " ;

	out << "0x" << std::hex << (unsigned)direction.get() << std::dec ;
	out << " => (z=" << direction.getZ() << ", y=" << direction.getY() ;
	out << ", x=" << direction.getX() << ")" ;

	out << "}" ;

	return out ;
}



HD
INLINE
const char *
toString( const Direction & direction )
{
	switch ( direction.get() )
	{
		case O    : return "O"   ;
		case N    : return "N"   ;
		case S    : return "S"   ;
		case E    : return "E"   ;
		case W    : return "W"   ;
		case T    : return "T"   ;
		case B    : return "B"   ;
		case NE   : return "NE"  ;
		case NW   : return "NW"  ;
		case NT   : return "NT"  ;
		case NB   : return "NB"  ;
		case SE   : return "SE"  ;
		case SW   : return "SW"  ;
		case ST   : return "ST"  ;
		case SB   : return "SB"  ;
		case ET   : return "ET"  ;
		case EB   : return "EB"  ;
		case WT   : return "WT"  ;
		case WB   : return "WB"  ;
		case NET  : return "NET" ;
		case NWT  : return "NWT" ;
		case SET  : return "SET" ;
		case SWT  : return "SWT" ;
		case NEB  : return "NEB" ;
		case NWB  : return "NWB" ;
		case SEB  : return "SEB" ;
		case SWB  : return "SWB" ;
	}
	return "UNKNOWN" ;
}



}



#endif
