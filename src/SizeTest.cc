#include "gtest/gtest.h"
#include "Size.hpp"



using namespace microflow ;



TEST(Size, empty_constructor)
{
	Size size ;

	EXPECT_EQ( 0u, size.getWidth() ) ;
	EXPECT_EQ( 0u, size.getHeight() ) ;
	EXPECT_EQ( 0u, size.getDepth() ) ;
}



TEST(Size, constructor)
{
	Size size(10, 20, 30) ;

	EXPECT_EQ( 10u, size.getWidth() ) ;
	EXPECT_EQ( 20u, size.getHeight() ) ;
	EXPECT_EQ( 30u, size.getDepth() ) ;
}



TEST(Size, in_range_empty)
{
	Size size ;

	EXPECT_FALSE( size.areCoordinatesInLimits(0,0,0) ) ;
}



TEST(Size, in_range)
{
	Size size(1,1,1) ;

	EXPECT_TRUE( size.areCoordinatesInLimits(0,0,0) ) ;
	EXPECT_FALSE( size.areCoordinatesInLimits(1,0,0) ) ;
	EXPECT_FALSE( size.areCoordinatesInLimits(0,1,0) ) ;
	EXPECT_FALSE( size.areCoordinatesInLimits(0,0,1) ) ;
	EXPECT_FALSE( size.areCoordinatesInLimits(1,1,1) ) ;
}



TEST(Size, operatorEqual)
{
	Size s1(1,2,3), s2(1,2,3), s3(3,2,1) ;
	Size s4(1,2,1), s5(1,1,3), s6(3,2,3) ;

	EXPECT_TRUE ( s1 == s2 ) ;
	EXPECT_FALSE( s1 == s3 ) ;
	EXPECT_FALSE( s1 == s4 ) ;
	EXPECT_FALSE( s1 == s5 ) ;
	EXPECT_FALSE( s1 == s6 ) ;
}



TEST(Size, operatorNotEqual)
{
	Size s1(1,2,3), s2(1,2,3), s3(3,2,1) ;
	Size s4(1,2,1), s5(1,1,3), s6(3,2,3) ;

	EXPECT_FALSE( s1 != s2 ) ;
	EXPECT_TRUE ( s1 != s3 ) ;
	EXPECT_TRUE ( s1 != s4 ) ;
	EXPECT_TRUE ( s1 != s5 ) ;
	EXPECT_TRUE ( s1 != s6 ) ;
}

