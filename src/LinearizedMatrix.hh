#ifndef LINEARIZED_MATRIX_HH
#define LINEARIZED_MATRIX_HH



namespace microflow
{



#define LINEARIZED_MATRIX_BASE_TEMPLATE_NO_INLINE   \
template< class T, template<class> class Storage >  \
HD 



#define LINEARIZED_MATRIX_BASE_TEMPLATE             \
LINEARIZED_MATRIX_BASE_TEMPLATE_NO_INLINE           \
INLINE



#define LINEARIZED_MATRIX_BASE   LinearizedMatrixBase< T, Storage >


#define LINEARIZED_MATRIX_TEMPLATE_NO_INLINE               \
template< class T, template<class> class Storage >



#define LINEARIZED_MATRIX_TEMPLATE                  \
LINEARIZED_MATRIX_TEMPLATE_NO_INLINE                \
INLINE



#define LINEARIZED_MATRIX       \
LinearizedMatrix< T, Storage >



LINEARIZED_MATRIX_TEMPLATE
LINEARIZED_MATRIX::
LinearizedMatrix()
{
}



LINEARIZED_MATRIX_BASE_TEMPLATE
LINEARIZED_MATRIX_BASE::
LinearizedMatrixBase()
{
}



LINEARIZED_MATRIX_BASE_TEMPLATE
LINEARIZED_MATRIX_BASE::
LinearizedMatrixBase( LinearizedMatrixBase< T, StorageOnGPU > & linearizedMatrixGPU )
: data_( linearizedMatrixGPU.data_ ),
	size_( linearizedMatrixGPU.size_ )
{
}



LINEARIZED_MATRIX_BASE_TEMPLATE
const T & LINEARIZED_MATRIX_BASE::
getValue( const Coordinates & coordinates ) const
{
	size_t linearIndex = computeInternalIndex( coordinates ) ;

	return data_[ linearIndex ] ;
}



LINEARIZED_MATRIX_BASE_TEMPLATE
void LINEARIZED_MATRIX_BASE::
setValue( const Coordinates & coordinates, const T & value ) 
{
	size_t linearIndex = computeInternalIndex( coordinates ) ;

	data_[ linearIndex ] = value ;	
}



LINEARIZED_MATRIX_BASE_TEMPLATE
Size LINEARIZED_MATRIX_BASE::
getSize() const
{
	return size_ ;
}



LINEARIZED_MATRIX_BASE_TEMPLATE
T & LINEARIZED_MATRIX_BASE::
operator[] ( const Coordinates & coordinates )
{
	size_t linearIndex = computeInternalIndex( coordinates ) ;

	return data_[ linearIndex ] ;
}



LINEARIZED_MATRIX_TEMPLATE
typename LINEARIZED_MATRIX::Iterator LINEARIZED_MATRIX::
begin()
{
	return data_.begin() ;
}



LINEARIZED_MATRIX_TEMPLATE
typename LINEARIZED_MATRIX::ConstIterator LINEARIZED_MATRIX::
begin() const
{
	return data_.begin() ;
}



LINEARIZED_MATRIX_TEMPLATE
typename LINEARIZED_MATRIX::Iterator LINEARIZED_MATRIX::
end()
{
	return data_.end() ;
}



LINEARIZED_MATRIX_TEMPLATE
typename LINEARIZED_MATRIX::ConstIterator LINEARIZED_MATRIX::
end() const
{
	return data_.end() ;
}



LINEARIZED_MATRIX_TEMPLATE
void LINEARIZED_MATRIX::
resize( const Size & size, const T & defaultValue )
{
	size_ = size ;
	data_.resize( size.computeVolume(), defaultValue ) ;
}



LINEARIZED_MATRIX_TEMPLATE
void LINEARIZED_MATRIX::
resizeWithContent( const Size & newSize, const T & edgeValue )
{
	if (newSize != size_)
	{
		std::vector< T > oldData = data_ ;
		Size oldSize = size_ ;

		size_ = newSize ;
		data_.resize( newSize.computeVolume() ) ;
		std::fill( data_.begin(), data_.end(), edgeValue ) ;


		size_t finalWidth  = std::min( oldSize.getWidth () , size_.getWidth () ) ;
		size_t finalHeight = std::min( oldSize.getHeight() , size_.getHeight() ) ;
		size_t finalDepth  = std::min( oldSize.getDepth () , size_.getDepth () ) ;

		for (size_t z=0 ; z < finalDepth ; z++)
			for (size_t y=0 ; y < finalHeight ; y++)
				for (size_t x=0 ; x < finalWidth ; x++)
				{
					Coordinates c(x, y, z) ;
					size_t oldInternalIndex = BaseType::computeInternalIndex(c, oldSize) ;
					T oldValue = oldData[ oldInternalIndex ] ;
					setValue( c, oldValue ) ;
				}
	}
}



LINEARIZED_MATRIX_BASE_TEMPLATE
size_t LINEARIZED_MATRIX_BASE::
computeInternalIndex( const Coordinates & coordinates ) const
{
	return computeInternalIndex(coordinates, getSize() ) ;
}



LINEARIZED_MATRIX_BASE_TEMPLATE
size_t LINEARIZED_MATRIX_BASE::
computeInternalIndex( const Coordinates & coordinates, const Size & size ) const
{
	return linearizeXYZ( coordinates, size ) ;
}



LINEARIZED_MATRIX_TEMPLATE
bool LINEARIZED_MATRIX::
operator==( const LINEARIZED_MATRIX & linearizedMatrix ) const
{
	return ( linearizedMatrix.size_ == size_  &&
					 linearizedMatrix.data_ == data_ ) ;
}



LINEARIZED_MATRIX_TEMPLATE_NO_INLINE
template< template<class> class StorageSource >
INLINE
void LINEARIZED_MATRIX::
operator=( const LinearizedMatrix<T, StorageSource> & sourceMatrix )
{
	size_ = sourceMatrix.size_ ;
	data_ = sourceMatrix.data_ ; 
}



template< class T >
LinearizedMatrix< T, StorageInKernel >::
LinearizedMatrix( LinearizedMatrix< T, StorageOnGPU > & linearizedMatrixGPU )
: LinearizedMatrixBase< T, StorageInKernel >( linearizedMatrixGPU )
{
} ;



#undef LINEARIZED_MATRIX_BASE
#undef LINEARIZED_MATRIX_BASE_TEMPLATE

#undef LINEARIZED_MATRIX
#undef LINEARIZED_MATRIX_TEMPLATE



}



#endif
