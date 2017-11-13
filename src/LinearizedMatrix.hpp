#ifndef LINEARIZED_MATRIX_HPP
#define LINEARIZED_MATRIX_HPP



#include <vector>
#include <cassert>

#include "Size.hpp"
#include "Storage.hpp"
#include "cudaPrefix.hpp"



namespace microflow
{



// Matrix, where data is stored in single memory chunk.
// TODO: maybe we should have possibilty to change order of elements.

template< class T, template<class> class Storage >
class LinearizedMatrixBase
{
	public:

		HD_WARNING_DISABLE
		HD LinearizedMatrixBase() ;

		HD_WARNING_DISABLE
		HD LinearizedMatrixBase( 
			LinearizedMatrixBase< T, StorageOnGPU > & linearizedMatrixGPU ) ;

		HD_WARNING_DISABLE
		HD const T & getValue( const Coordinates & coordinates ) const ;

		HD_WARNING_DISABLE
		HD void setValue( const Coordinates & coordinates, const T & value ) ;

		HD_WARNING_DISABLE
		HD T & operator[](const Coordinates & coordinates) ;

		HD Size getSize() const ;

		// Required for calls of optimized kernels.
		T * getDataPointer() { return data_.getPointer() ; }


	protected:

		HD size_t computeInternalIndex( const Coordinates & coordinates ) const ;
		HD size_t computeInternalIndex( const Coordinates & coordinates, const Size & size ) const ;

		template< class, template<class> class> friend class LinearizedMatrixBase ;

		Storage<T> data_ ;
		Size       size_ ;
} ;



#define USE_LINEARIZED_MATRIX_BASE_METHODS \
	using BaseType::getValue ;               \
	using BaseType::setValue ;               \
	using BaseType::operator[] ;             \
	using BaseType::getSize ;



template< class T, template<class> class Storage = StorageOnCPU >
class LinearizedMatrix
: public LinearizedMatrixBase< T, Storage >
{
	private: typedef LinearizedMatrixBase< T, Storage >  BaseType ;


	public:

		LinearizedMatrix() ;

		
		USE_LINEARIZED_MATRIX_BASE_METHODS


		void resize( const Size & size, const T & defaultValue = 0 ) ;
		void resizeWithContent( const Size & newSize, const T & edgeValue = 0 ) ;

		typedef typename Storage<T>::iterator Iterator;
    typedef typename Storage<T>::const_iterator ConstIterator;
    
		Iterator begin() ;
    ConstIterator begin() const ;
    Iterator end() ;
    ConstIterator end() const ;

		template< template<class> class StorageSource >
		void operator=(const LinearizedMatrix<T, StorageSource> & sourceMatrix ) ;

		bool operator==( const LinearizedMatrix & linearizedMatrix ) const ;


	private:

		using BaseType::data_ ;
		using BaseType::size_ ;

		template< class, template<class> class> friend class LinearizedMatrix ;
} ;



template< class T >
class LinearizedMatrix< T, StorageInKernel >
: public LinearizedMatrixBase< T, StorageInKernel >
{
	private: typedef LinearizedMatrixBase< T, StorageInKernel >  BaseType ;


	public:

		LinearizedMatrix( LinearizedMatrix< T, StorageOnGPU > & linearizedMatrixGPU ) ;

		
		USE_LINEARIZED_MATRIX_BASE_METHODS


	private:

		using BaseType::data_ ;
		using BaseType::size_ ;
} ;



#undef USE_LINEARIZED_MATRIX_BASE_METHODS



}



#include "LinearizedMatrix.hh"



#endif
