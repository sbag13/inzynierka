#include "gtest/gtest.h"
#include "LatticeArrangementD3Q27.hpp"



#include <iostream>



using namespace microflow ;



#define DISABLE_COMPILER_WARNING(d) (void)d ;



TEST( LatticeArrangementD3Q27, numberOfC )
{
	unsigned counter = 0 ;
	
	for ( auto d : D3Q27::c)
	{
		counter ++ ;
		DISABLE_COMPILER_WARNING(d) ;
	}

	EXPECT_EQ( 27u, counter ) ;
}



TEST( LatticeArrangementD3Q27, getIndex )
{
	unsigned i=0 ;

	for ( auto d : D3Q27::c )
	{
		unsigned directionIndex = D3Q27::getIndex(d) ;

		EXPECT_EQ( d, D3Q27::c[ directionIndex ] ) 
			<< " i = " << i
			<< ", in direction : " << Direction(d) 
			<< ", index = " << directionIndex 
			<< ", out direction : " << Direction(D3Q27::c[ directionIndex ]) << "\n" ;

		i++ ;
	}
}



TEST( LatticeArrangementD3Q27, getName )
{
	std::cout << D3Q27::getName() << "\n" ;
	EXPECT_EQ( "D3Q27", D3Q27::getName() ) ;
}



