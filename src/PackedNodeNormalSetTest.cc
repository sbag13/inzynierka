#include "gtest/gtest.h"
#include "PackedNodeNormalSet.hpp"



using namespace microflow ;
using namespace std ;



TEST( PackedNodeNormalSet, sizeof )
{
	EXPECT_EQ( sizeof(PackedNodeNormalSet), 8u ) ;
}



TEST( PackedNodeNormalSet, getNormalVector_assert )
{
	PackedNodeNormalSet p ;

	ASSERT_NO_THROW( p.getNormalVector(0) ) ;
	ASSERT_NO_THROW( p.getNormalVector(1) ) ;
	ASSERT_NO_THROW( p.getNormalVector(2) ) ;
	ASSERT_NO_THROW( p.getNormalVector(3) ) ;
	ASSERT_NO_THROW( p.getNormalVector(4) ) ;
	ASSERT_NO_THROW( p.getNormalVector(5) ) ;
	ASSERT_NO_THROW( p.getNormalVector(6) ) ;

	ASSERT_DEATH( p.getNormalVector(7), "" ) ;
}



TEST(PackedNodeNormalSet, emptyConstructor )
{
	PackedNodeNormalSet p ;

	EXPECT_EQ( 0u, p.getNormalVectorsCounter() ) ;
	EXPECT_EQ( 0u, p.getEdgeNodeType() ) ;

	EXPECT_EQ( 0, p.getNormalVector(0).get() ) ;
	EXPECT_EQ( 0, p.getNormalVector(1).get() ) ;
	EXPECT_EQ( 0, p.getNormalVector(2).get() ) ;
	EXPECT_EQ( 0, p.getNormalVector(3).get() ) ;
	EXPECT_EQ( 0, p.getNormalVector(4).get() ) ;
	EXPECT_EQ( 0, p.getNormalVector(5).get() ) ;
	EXPECT_EQ( 0, p.getNormalVector(6).get() ) ;
}



TEST(PackedNodeNormalSet, addNormalVector_assert )
{
	PackedNodeNormalSet p ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH ) ) ;
	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::EAST  ) ) ;
	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::WEST  ) ) ;
	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::TOP   ) ) ;
	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::BOTTOM) ) ;
	ASSERT_NO_THROW( p.addNormalVector( Direction::SOUTH + Direction::EAST  ) ) ;

	ASSERT_ANY_THROW( p.addNormalVector( Direction::SOUTH + Direction::WEST  ) ) ;
}



TEST( PackedNodeNormalSet, addNormalVector_WEST )
{
	PackedNodeNormalSet p ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::WEST ) ) ;
	EXPECT_EQ( 1u, p.getNormalVectorsCounter() ) ;
	EXPECT_EQ( Direction::WEST, p.getNormalVector(0).get() ) ;
}



TEST( PackedNodeNormalSet, addNormalVector_SOUTH )
{
	PackedNodeNormalSet p ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::SOUTH ) ) ;
	EXPECT_EQ( 1u, p.getNormalVectorsCounter() ) ;
	EXPECT_EQ( Direction::SOUTH, p.getNormalVector(0).get() ) ;
}



TEST( PackedNodeNormalSet, addNormalVector_BOTTOM )
{
	PackedNodeNormalSet p ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::BOTTOM ) ) ;
	EXPECT_EQ( 1u, p.getNormalVectorsCounter() ) ;
	EXPECT_EQ( Direction::BOTTOM, p.getNormalVector(0).get() ) ;
}



TEST( PackedNodeNormalSet, addNormalVector )
{
	PackedNodeNormalSet p ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH ) ) ;
	EXPECT_EQ( 1u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH ) ) ;
	EXPECT_EQ( 1u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::SOUTH ) ) ;
	EXPECT_EQ( 2u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH ) ) ;
	EXPECT_EQ( 2u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::EAST) ) ;
	EXPECT_EQ( 3u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::SOUTH ) ) ;
	EXPECT_EQ( 3u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::WEST) ) ;
	EXPECT_EQ( 4u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::TOP) ) ;
	EXPECT_EQ( 5u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::EAST) ) ;
	EXPECT_EQ( 5u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::SOUTH ) ) ;
	EXPECT_EQ( 5u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::WEST) ) ;
	EXPECT_EQ( 5u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::NORTH + Direction::TOP) ) ;
	EXPECT_EQ( 5u, p.getNormalVectorsCounter() ) ;

	ASSERT_NO_THROW( p.addNormalVector( Direction::BOTTOM) ) ;
	EXPECT_EQ( 6u, p.getNormalVectorsCounter() ) ;

	EXPECT_EQ( Direction::NORTH, p.getNormalVector(0).get() ) ;
	EXPECT_EQ( Direction::SOUTH, p.getNormalVector(1).get() ) ;
	EXPECT_EQ( Direction::NORTH + Direction::EAST, p.getNormalVector(2).get() ) ;
	EXPECT_EQ( Direction::NORTH + Direction::WEST, p.getNormalVector(3).get() ) ;
	EXPECT_EQ( Direction::NORTH + Direction::TOP , p.getNormalVector(4).get() ) ;
	EXPECT_EQ( Direction::BOTTOM, p.getNormalVector(5).get() ) ;
}



TEST( PackedNodeNormalSet, calculateResultantNormalVector )
{
	PackedNodeNormalSet p ;

	p.calculateResultantNormalVector() ;

	EXPECT_EQ( Coordinates(0), p.getResultantNormalVector() ) ;

	p.addNormalVector( Direction::EAST ) ;
	p.calculateResultantNormalVector() ;
	EXPECT_EQ( Direction::EAST, p.getResultantNormalVector().get() ) ;
	EXPECT_EQ( Direction::EAST, p.getNormalVector(0).get() ) ;

	p.addNormalVector( Direction::NORTH ) ;
	p.calculateResultantNormalVector() ;
	EXPECT_EQ( Direction::EAST + Direction::NORTH, p.getResultantNormalVector().get() ) ;
	EXPECT_EQ( Direction::EAST , p.getNormalVector(0).get() ) ;
	EXPECT_EQ( Direction::NORTH, p.getNormalVector(1).get() ) ;

	p.addNormalVector( Direction::TOP ) ;
	p.calculateResultantNormalVector() ;
	EXPECT_EQ( Direction::EAST + Direction::NORTH + Direction::TOP, 
						 p.getResultantNormalVector().get() ) ;
	EXPECT_EQ( Direction::EAST , p.getNormalVector(0).get() ) ;
	EXPECT_EQ( Direction::NORTH, p.getNormalVector(1).get() ) ;
	EXPECT_EQ( Direction::TOP  , p.getNormalVector(2).get() ) ;
}
