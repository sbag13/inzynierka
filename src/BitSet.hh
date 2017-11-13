#ifndef BITSET_HH
#define BITSET_HH



#include "Exceptions.hpp"



namespace microflow
{



HD INLINE
void BitSet::
clear()
{
	set_ = 0 ;
}


		
HD INLINE
bool BitSet::
test (size_t pos) const
{
	ASSERT (pos < 32) ;

	return ( 0 != ( mask(pos) & set_ ) ) ;
}



HD INLINE
void BitSet::
set (size_t pos)
{
	ASSERT (pos < 32) ;

	set_ |= mask(pos) ;
}



HD INLINE
unsigned int BitSet::
mask (size_t pos)
{
	return (1 << pos) ;
}



HD INLINE
BitSet BitSet::
operator| (const BitSet & arg) const
{
	BitSet result ;
	result.set_ = this->set_ | arg.set_ ;
	return result ;
}



HD INLINE
bool BitSet::
operator== (const BitSet & arg) const
{
	return ( this->set_ == arg.set_ ) ;
}



inline
std::ostream & operator<< (std::ostream & out, const BitSet & bitSet)
{
	for (unsigned pos=0 ; pos < 32 ; pos++)
	{
		if ( bitSet.test(pos) )
		{
			out << '1' ;
		}
		else
		{
			out << '0' ;
		}
	}

	return out ;
}



}



#endif
