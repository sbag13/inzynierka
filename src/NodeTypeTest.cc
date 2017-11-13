#include "gtest/gtest.h"

#include "NodeType.hpp"



using namespace microflow ;



TEST( NodeType, size )
{
	EXPECT_EQ( 2u, sizeof(NodeType) ) ;
}



TEST( NodeType, sizeInBits )
{
	EXPECT_EQ( 0u, sizeInBits(0) ) ;
	
	EXPECT_EQ( 1u, sizeInBits(1) ) ;
	
	EXPECT_EQ( 2u, sizeInBits(2) ) ;
	EXPECT_EQ( 2u, sizeInBits(3) ) ;
	
	EXPECT_EQ( 3u, sizeInBits(4) ) ;
	EXPECT_EQ( 3u, sizeInBits(5) ) ;
	EXPECT_EQ( 3u, sizeInBits(6) ) ;
	EXPECT_EQ( 3u, sizeInBits(7) ) ;

	EXPECT_EQ( 4u, sizeInBits(8) ) ;
	EXPECT_EQ( 4u, sizeInBits(9) ) ;
	EXPECT_EQ( 4u, sizeInBits(15) ) ;

	EXPECT_EQ( 5u, sizeInBits(16) ) ;
	EXPECT_EQ( 5u, sizeInBits(17) ) ;
	EXPECT_EQ( 5u, sizeInBits(31) ) ;

	EXPECT_EQ( 6u, sizeInBits(32) ) ;
	EXPECT_EQ( 6u, sizeInBits(33) ) ;
	EXPECT_EQ( 6u, sizeInBits(63) ) ;

	EXPECT_EQ( 7u, sizeInBits(64) ) ;
	EXPECT_EQ( 7u, sizeInBits(65) ) ;
	EXPECT_EQ( 7u, sizeInBits(127) ) ;
}



TEST( NodeType, sizeInBits_nodeBaseType )
{
	EXPECT_EQ( 3u, sizeInBits(static_cast<unsigned>(NodeBaseType::SIZE)) ) ;
}



TEST( NodeType, compare )
{
	NodeType node1(NodeBaseType::SOLID) ;
	NodeType node2(NodeBaseType::SOLID) ;
	NodeType node3(NodeBaseType::FLUID) ;
	NodeType node4(NodeBaseType::FLUID) ;

	EXPECT_TRUE  (node1 == node2) ;
	EXPECT_FALSE (node1 != node2) ;
	EXPECT_FALSE (node1 == node3) ;
	EXPECT_TRUE  (node1 != node3) ;

	EXPECT_TRUE  (node3 == node4) ;
	EXPECT_FALSE (node3 != node4) ;

	NodeType node5, node6 ;
	
	node5.setBaseType( NodeBaseType::FLUID ) ;
	node6.setBaseType( NodeBaseType::FLUID ) ;

	EXPECT_TRUE  (node5 == node6) ;
	EXPECT_FALSE (node5 != node6) ;
}



TEST( PlacementModifierNames, stringToKey )
{
	EXPECT_EQ (PlacementModifier::NONE   , fromString<PlacementModifier> ("none"  ) ) ;
	EXPECT_EQ (PlacementModifier::NORTH  , fromString<PlacementModifier> ("north" ) ) ;
	EXPECT_EQ (PlacementModifier::SOUTH  , fromString<PlacementModifier> ("south" ) ) ;
	EXPECT_EQ (PlacementModifier::EAST   , fromString<PlacementModifier> ("east"  ) ) ;
	EXPECT_EQ (PlacementModifier::WEST   , fromString<PlacementModifier> ("west"  ) ) ;
	EXPECT_EQ (PlacementModifier::BOTTOM , fromString<PlacementModifier> ("bottom") ) ;
	EXPECT_EQ (PlacementModifier::TOP    , fromString<PlacementModifier> ("top"   ) ) ;

	EXPECT_EQ (PlacementModifier::EXTERNAL_EDGE,
			fromString<PlacementModifier> ("external_edge") ) ;    
	EXPECT_EQ (PlacementModifier::EXTERNAL_EDGE_PRESSURE_TANGENTIAL,
			fromString<PlacementModifier> ("external_edge_pressure_tangential") ) ;    
	EXPECT_EQ (PlacementModifier::INTERNAL_EDGE,
			fromString<PlacementModifier> ("internal_edge") ) ;    
	EXPECT_EQ (PlacementModifier::EXTERNAL_CORNER,
			fromString<PlacementModifier> ("external_corner") ) ;    
	EXPECT_EQ (PlacementModifier::EXTERNAL_CORNER_PRESSURE_TANGENTIAL,
			fromString<PlacementModifier> ("external_corner_pressure_tangential") ) ;    
	EXPECT_EQ (PlacementModifier::CORNER_ON_EDGE_AND_PERPENDICULAR_PLANE,
			fromString<PlacementModifier> ("corner_on_edge_and_perpendicular_plane") ) ;    
}



