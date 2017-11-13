#ifndef STORAGE_HH
#define STORAGE_HH



#include "Logger.hpp"
#include "TestTools.hpp"



namespace microflow
{



template< class T >
template< class U >
T & StorageOnCPU<T>::
operator[] ( U idx )
{
	return std::vector<T>::operator[] ( idx ) ;
}



template< class T >
template< class U >
const T & StorageOnCPU<T>:: template
operator[] ( U idx ) const
{
	return std::vector<T>::operator[] ( idx ) ;
}



namespace
{

	template< template<class> class Storage1,
						template<class> class Storage2,
						class T>
	void
	copy( Storage1<T> & destination, const Storage2<T> & source )
	{
		destination.resize( source.size() ) ;
		thrust::copy( source.begin(), source.end(), destination.begin() ) ;
	}

}



template<class T>
template< template<class> class Storage >
inline
StorageOnCPU<T> & StorageOnCPU<T>::
operator=( const Storage<T> & rhs )
{
	copy( *this, rhs ) ;
	return *this ;
}



template< class T >
StorageOnGPU<T>::
StorageOnGPU()
: thrust::device_vector<T>()
{
}



template<class T>
template< template<class> class Storage >
inline
StorageOnGPU<T> & StorageOnGPU<T>::
operator=( const Storage<T> & rhs )
{
	copy( *this, rhs ) ;
	return *this ;
}



template< class T >
T * StorageOnGPU<T>::
getPointer()
{
	return thrust::raw_pointer_cast( this->data() ) ;
}



template< class T >
size_t StorageOnGPU<T>::
getNumberOfElements()
{
	return this->size() ;
}



template< class T >
bool differ( const T & lhs, const T & rhs ) 
{
	return lhs != rhs ;
}



template<>
inline
bool differ( const double & lhs, const double & rhs )
{
	return areFloatsNotEqual( lhs, rhs ) ;
}



template< template<class> class Storage1, 
					template<class> class Storage2,
					class T >
bool operator==( const Storage1<T> & lhs, const Storage2<T> & rhs )
{
	if ( lhs.size() != rhs.size() )
	{
		logger << "lhs.size()=" << lhs.size() << " differs from rhs.size()=" 
					 << rhs.size() << "\n" ;
		return false ;
	}

	for (unsigned i=0 ; i < lhs.size() ; i++)
	{
		T r = rhs[i] ;
		T l = lhs[i] ;
		if ( differ( l, r ) )
		{
			logger << "lhs[" << i << "]=" << lhs[i] << " differs from "
						 << "rhs[" << i << "]=" << rhs[i] << "\n" ;
			return false ;
		}
	}

	return true ;
}



template<class T>
StorageInKernel<T>::
StorageInKernel( StorageOnGPU<T> & storageOnGPU )
: dataPointer_     ( storageOnGPU.getPointer() ),
	numberOfElements_( storageOnGPU.getNumberOfElements() )
{
}



template< class T >
template< class U >
inline
HD T & StorageInKernel<T>::
operator[] ( U idx )
{
	return dataPointer_[ idx ] ;
}



template< class T >
template< class U >
inline
HD const T & StorageInKernel<T>::
operator[] ( U idx ) const
{
	return dataPointer_[ idx ] ;
}



template< class T >
inline
HD size_t StorageInKernel<T>::
size() const
{
	return numberOfElements_ ;
}



}



#endif
