#ifndef PACKED_DIRECTION_VECTOR_HH
#define PACKED_DIRECTION_VECTOR_HH



#include <cstddef>
#include "Exceptions.hpp"



namespace microflow
{



template<class T>
inline
PackedDirectionVector<T>::
PackedDirectionVector()
: packedVector_(0)
{
}



template<class T>
inline
PackedDirectionVector<T>::
PackedDirectionVector(PackedDirectionVector<T>::InternalStorageType packedVector)
: packedVector_( packedVector ) 
{
}



template<class T>
inline
typename PackedDirectionVector<T>::InternalStorageType PackedDirectionVector<T>::
extractSignedBits( unsigned positionLSB, unsigned noOfBits ) const 
{
	PackedDirectionVector<T>::InternalStorageType tmp = packedVector_ ;

	constexpr size_t sizeOf = sizeof(tmp) * 8 ;

	tmp <<= sizeOf - (positionLSB + noOfBits ) ;
	tmp >>= sizeOf - noOfBits ;

	return tmp ;	
}



template<class T>
inline
typename PackedDirectionVector<T>::InternalStorageType PackedDirectionVector<T>::
extractUnsignedBits( unsigned positionLSB, unsigned noOfBits ) const 
{
	PackedDirectionVector<T>::InternalStorageType tmp ;

	tmp = extractSignedBits(positionLSB, noOfBits) ;
	
	PackedDirectionVector<T>::InternalStorageType mask ;
	
	mask = (1 << noOfBits) - 1 ;

	return tmp & mask ;	
}



template<class T>
inline
char PackedDirectionVector<T>::
extractSigned2Bits( unsigned positionLSB ) const 
{
	return extractSignedBits( positionLSB, 2 ) ;
}



template<class T>
inline
char PackedDirectionVector<T>::
getCoordinate( unsigned index ) const
{
	return extractSigned2Bits( 2 * index ) ;
}



template<class T>
inline
void PackedDirectionVector<T>::
setCoordinate( unsigned index, char value /* -1, 0, +1  only */ )
{
	ASSERT(value >= -1) ;
	ASSERT(value <=  1) ;

	unsigned positionLSB = index * 2 ;

	PackedDirectionVector<T>::InternalStorageType andMask ;
	andMask = ~(3 << positionLSB) ;

	PackedDirectionVector<T>::InternalStorageType shiftedValue ;
	shiftedValue = (value & 3) << positionLSB ;

	packedVector_ &= andMask ;
	packedVector_ |= shiftedValue ;
}



template<class T>
inline
char PackedDirectionVector<T>::
getVector3D( unsigned vectorIndex ) const
{
	return extractUnsignedBits( bitsPerVector3D * vectorIndex, bitsPerVector3D ) ;
}



template<class T>
template<class V>
inline
void PackedDirectionVector<T>::
setVector3D( unsigned vectorIndex, V vector3D )
{
	const unsigned positionLSB = vectorIndex * bitsPerVector3D ;

	const InternalStorageType andMask = 
					static_cast<InternalStorageType>(bitMaskVector3D) << positionLSB ;

	packedVector_ &= ( ~ andMask) ; // make place for vector

	InternalStorageType vectorBits = vector3D.get() & 0b00111111 ;
	vectorBits <<= positionLSB ;

	packedVector_ |= vectorBits ;
}



template<class T>
inline
typename PackedDirectionVector<T>::InternalStorageType PackedDirectionVector<T>::
getInternalStorage() const
{
	return packedVector_ ;
}



}



#endif
