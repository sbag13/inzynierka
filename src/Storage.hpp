#ifndef STORAGE_HPP
#define STORAGE_HPP



#include <vector>
#include <string>
// FIXME: warnings !!! Make all files cu ? Then CUDA required for compilation (many ifdefs)...
// TODO: Replace thrust with cub ? Check http://nvlabs.github.io/cub/index.html
//#include <thrust/host_vector.h> 
#include <thrust/device_vector.h>



#include "cudaPrefix.hpp"



namespace microflow
{



/*
	Used to manage CPU memory. Can copy data from StorageOnGPU objects.
*/
template<class T>
class StorageOnCPU : public std::vector<T>
{
	public:
		template< template<class> class Storage >
		StorageOnCPU & operator=(const Storage<T> & rhs) ;

		template<class U>	
		T & operator[] ( U idx ) ;

		template<class U>	
		const T & operator[] ( U idx ) const ;

		static const std::string getName() { return "StorageOnCPU" ; }
} ;



/*
	Used to manage GPU memory from host code. Can copy data from StorageOnCPU objects.
*/
template<class T>
class StorageOnGPU : public thrust::device_vector<T>
{
	public:

		//TODO: check, where it is called, without this have warning host device.
		StorageOnGPU() ;

		template< template<class> class Storage >
		StorageOnGPU & operator=(const Storage<T> & rhs) ;

		T * getPointer() ; //TODO: make this protected and friend classes ?
		size_t getNumberOfElements() ;

		static const std::string getName() { return "StorageOnGPU" ; }
} ;



template< template <class> class Storage1, 
					template <class> class Storage2,
					class T >
bool operator==( const Storage1<T> & lhs, const Storage2<T> & rhs ) ;



/*
	Used to wrap pointers while passing StorageOnGPU objects to kernels. 
	Automatically extracts pointers from StorageOnGPU.
*/
template<class T>
class StorageInKernel
{
	public:

		StorageInKernel( StorageOnGPU<T> & storageOnGPU ) ;

		typedef size_t iterator ;
		typedef size_t const_iterator ;
		
		template<class U>
		HD T & operator[] (U idx) ;

		template<class U>
		HD const T & operator[] (U idx) const ;
		
		HD size_t size() const ;

	private:
		T * dataPointer_ ;
		size_t numberOfElements_ ;
} ;



}



#include "Storage.hh"



#endif
