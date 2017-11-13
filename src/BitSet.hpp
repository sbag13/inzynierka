#ifndef BITSET_HPP
#define BITSET_HPP



#include <ostream>

#include "cudaPrefix.hpp"



namespace microflow
{

/*
	Replacement for std::bitset, which has no CUDA implementation.
*/
//TODO: template< capacity >
class BitSet
{
	public:

		HD void clear () ;
		HD bool test (size_t pos) const ;
		HD void set (size_t pos) ;
		HD BitSet operator| (const BitSet & arg) const ;
		HD bool operator== (const BitSet & arg ) const ;
		HD bool isClear() const { return 0 == set_ ; }
	
	private:
		
		HD static unsigned int mask (size_t pos) ;

		unsigned int set_ ;
} ;



std::ostream& operator<< (std::ostream& out, const BitSet & bitSet) ;



}



#include "BitSet.hh"



#endif
