#include "gtest/gtest.h"
#include "NodeBaseType.hpp"



using namespace microflow ;
using namespace std ;



TEST( NodeBaseTypeNames, keyToString )
{
	EXPECT_EQ( "solid"        , toString(NodeBaseType::SOLID) ) ;
	EXPECT_EQ( "fluid"        , toString(NodeBaseType::FLUID) ) ;
	EXPECT_EQ( "bounce_back_2", toString(NodeBaseType::BOUNCE_BACK_2) ) ;
	EXPECT_EQ( "velocity"     , toString(NodeBaseType::VELOCITY) ) ;
	EXPECT_EQ( "velocity_0"   , toString(NodeBaseType::VELOCITY_0) ) ;
	EXPECT_EQ( "pressure"     , toString(NodeBaseType::PRESSURE) ) ;
}



TEST( NodeBaseTypeNames, stringToKey )
{
	EXPECT_EQ( NodeBaseType::SOLID         , fromString<NodeBaseType>("solid"        ) ) ;
	EXPECT_EQ( NodeBaseType::FLUID         , fromString<NodeBaseType>("fluid"        ) ) ;
	EXPECT_EQ( NodeBaseType::BOUNCE_BACK_2 , fromString<NodeBaseType>("bounce_back_2") ) ;
	EXPECT_EQ( NodeBaseType::VELOCITY      , fromString<NodeBaseType>("velocity"     ) ) ;
	EXPECT_EQ( NodeBaseType::VELOCITY_0    , fromString<NodeBaseType>("velocity_0"   ) ) ;
	EXPECT_EQ( NodeBaseType::PRESSURE      , fromString<NodeBaseType>("pressure"     ) ) ;
}



TEST( NodeBaseTypeNames, stringToKeyOutOfRange )
{
	EXPECT_THROW( fromString<NodeBaseType>("strange name not used"), std::out_of_range ) ;
}



