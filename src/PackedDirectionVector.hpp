#ifndef PACKED_DIRECTION_VECTOR_HPP
#define PACKED_DIRECTION_VECTOR_HPP



#include "cudaPrefix.hpp"



namespace microflow
{



template<class T>
class PackedDirectionVector
{
	public:

		typedef T InternalStorageType ;

		HD PackedDirectionVector<T>() ;
		HD PackedDirectionVector<T>( PackedDirectionVector<T>::InternalStorageType packedVector ) ;
                              		
		// Treats internal data as an indexed set of 2-bit fields
		HD char getCoordinate( unsigned index ) const ; 
		HD void setCoordinate( unsigned index, char value /* -1, 0, +1  only */ ) ;
		// Treats internal data as an indexed set of 3*2 = 6-bit fields
		HD char getVector3D( unsigned vectorIndex ) const ;

		template<class V>
		HD void setVector3D( unsigned vectorIndex, V vector3D) ;

		HD InternalStorageType getInternalStorage() const ;


	private:

		HD InternalStorageType extractSignedBits( unsigned positionLSB, unsigned noOfBits ) const ;
		HD InternalStorageType extractUnsignedBits( unsigned positionLSB, unsigned noOfBits ) const ;
		HD char extractSigned2Bits( unsigned positionLSB ) const ;

		static constexpr unsigned bitsPerVector3D = 3*2 ;
		static constexpr unsigned bitMaskVector3D = 0b00111111 ;

		T packedVector_ ;
} ;



}



#include "PackedDirectionVector.hh"



#endif
