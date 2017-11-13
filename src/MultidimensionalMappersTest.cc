#include "gtest/gtest.h"
#include "MultidimensionalMappers.hpp"



using namespace microflow ;
using namespace std ;



TEST (MultidimensionalMappers, XYZ_linearize)
{
	const unsigned xMax = 4 ;
	const unsigned yMax = 5 ;
	const unsigned zMax = 6 ;


	unsigned linearIndex = 0 ;

	for (unsigned z=0 ; z < zMax ; z++)
		for (unsigned y=0 ; y < yMax ; y++)
			for (unsigned x=0 ; x < xMax ; x++)
			{
				EXPECT_EQ (linearIndex, XYZ::linearize (x,y,z, xMax, yMax, zMax))
					<< " x=" << x << ", y=" << y << ", z= " << z << "\n" ;

				linearIndex++ ;
			}
}



TEST (MultidimensionalMappers, YXZ_linearize)
{
	const unsigned xMax = 4 ;
	const unsigned yMax = 5 ;
	const unsigned zMax = 6 ;


	unsigned linearIndex = 0 ;

	for (unsigned z=0 ; z < zMax ; z++)
		for (unsigned x=0 ; x < xMax ; x++)
			for (unsigned y=0 ; y < yMax ; y++)
			{
				EXPECT_EQ (linearIndex, YXZ::linearize (x,y,z, xMax, yMax, zMax))
					<< " x=" << x << ", y=" << y << ", z= " << z << "\n" ;

				linearIndex++ ;
			}
}



TEST (MultidimensionalMappers, ZigzagNE_linearize)
{
	// WARNING: ZigzagNE works only for 4x4x4 tiles.
	const unsigned xMax = 4 ;
	const unsigned yMax = 4 ;
	const unsigned zMax = 4 ;


#define TEST_ZIGZAG_NE_LINEARIZE(x,y,z, index) \
	EXPECT_EQ (index, ZigzagNE::linearize (x,y,z, xMax, yMax, zMax)) \
			<< " x=" << x << ", y=" << y << ", z= " << z << "\n" ;

	for (unsigned z=0 ; z < 4 ; z++)
	{
		TEST_ZIGZAG_NE_LINEARIZE( 0,0,z,   0u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 1,0,z,   1u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 2,0,z,   2u * 2 + (z&1) + (z&2) * 16)

		TEST_ZIGZAG_NE_LINEARIZE( 0,1,z,   3u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 1,1,z,   4u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 2,1,z,   5u * 2 + (z&1) + (z&2) * 16)

		TEST_ZIGZAG_NE_LINEARIZE( 0,2,z,   6u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 1,2,z,   7u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 2,2,z,   8u * 2 + (z&1) + (z&2) * 16)

		TEST_ZIGZAG_NE_LINEARIZE( 0,3,z,   9u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 1,3,z,  10u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 2,3,z,  11u * 2 + (z&1) + (z&2) * 16)

		TEST_ZIGZAG_NE_LINEARIZE( 3,3,z,  12u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 3,2,z,  13u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 3,1,z,  14u * 2 + (z&1) + (z&2) * 16)
		TEST_ZIGZAG_NE_LINEARIZE( 3,0,z,  15u * 2 + (z&1) + (z&2) * 16)
	}

#undef TEST_ZIGZAG_NE_LINEARIZE
}
