#include "gtest/gtest.h"
#include "ColorAssignment.hpp"
#include <sstream>

using namespace microflow ;
using namespace std ;



TEST( ColorAssignment, emptyConstructor )
{
	ColorAssignment colorAssignment ;

	EXPECT_EQ( true , colorAssignment.isSolid() ) ;
	EXPECT_EQ( false, colorAssignment.isFluid() ) ;
	EXPECT_EQ( false, colorAssignment.isBoundary() ) ;
	EXPECT_EQ( false, colorAssignment.isCharacteristicLengthMarker() ) ;
}



TEST( ColorAssignment, read_general )
{
	stringstream ss ;

	ss << "# \n" ;
	ss << "{\n" ;
	ss << "# node_type \n" ;
	ss << "# some comment \n" ;
	ss << "# \n" ;
	ss << "node_type = solid\n" ;
	ss << "velocity = (1.0, 2, 3.5)\n" ;
	ss << "pressure = 5.5\n" ;
	ss << "color  =  (128,129,130) \n" ;
	ss << "}\n" ;

	ColorAssignment colorAssignment ;

	EXPECT_NO_THROW( ss >> colorAssignment ) ;

	EXPECT_TRUE( colorAssignment.isSolid() ) ;
	EXPECT_EQ( NodeBaseType::SOLID, colorAssignment.getNodeBaseType() ) ;

	EXPECT_EQ( 1.0, colorAssignment.getVelocity()[0] ) ;
	EXPECT_EQ( 2.0, colorAssignment.getVelocity()[1] ) ;
	EXPECT_EQ( 3.5, colorAssignment.getVelocity()[2] ) ;

	EXPECT_EQ( 5.5, colorAssignment.getPressure() ) ;

	EXPECT_TRUE( colorAssignment.colorEquals( png::rgb_pixel(128,129,130) ) ) ;
	EXPECT_FALSE( colorAssignment.colorEquals( png::rgb_pixel(1,1,1) ) ) ;
}



TEST( ColorAssignment, read_node_type_fluid )
{
	stringstream ss ;

	ss << "{\n" ;
	ss << "node_type = fluid\n" ;
	ss << "}\n" ;

	ColorAssignment colorAssignment ;

	EXPECT_NO_THROW( ss >> colorAssignment ) ;

	EXPECT_TRUE ( colorAssignment.isFluid() ) ;
	EXPECT_FALSE( colorAssignment.isSolid() ) ;
	EXPECT_FALSE( colorAssignment.isBoundary() ) ;
	EXPECT_FALSE( colorAssignment.isCharacteristicLengthMarker() ) ;

	EXPECT_EQ( NodeBaseType::FLUID, colorAssignment.getNodeBaseType() ) ;
}



TEST( ColorAssignment, read_node_type_marker_true )
{
	stringstream ss ;

	ss << "{\n" ;
	ss << "characteristic_length_marker = true\n" ;
	ss << "}\n" ;

	ColorAssignment colorAssignment ;

	EXPECT_NO_THROW( ss >> colorAssignment ) ;

	EXPECT_FALSE( colorAssignment.isFluid() ) ;
	EXPECT_FALSE( colorAssignment.isBoundary() ) ;

	EXPECT_TRUE ( colorAssignment.isCharacteristicLengthMarker() ) ;
	EXPECT_TRUE ( colorAssignment.isSolid() ) ;
}



TEST( ColorAssignment, read_node_type_marker_false )
{
	stringstream ss ;

	ss << "{\n" ;
	ss << "characteristic_length_marker = false\n" ;
	ss << "}\n" ;

	ColorAssignment colorAssignment ;

	EXPECT_NO_THROW( ss >> colorAssignment ) ;

	EXPECT_FALSE( colorAssignment.isFluid() ) ;
	EXPECT_FALSE( colorAssignment.isBoundary() ) ;

	EXPECT_FALSE( colorAssignment.isCharacteristicLengthMarker() ) ;
	EXPECT_TRUE ( colorAssignment.isSolid() ) ;
}



TEST( ColorAssignment, read_node_type_marker_exception )
{
	stringstream ss ;

	ss << "{\n" ;
	ss << "characteristic_length_marker = 0\n" ;
	ss << "}\n" ;

	ColorAssignment colorAssignment ;

	EXPECT_ANY_THROW( ss >> colorAssignment ) ;
}



TEST( ColorAssignment, comment_without_space_first_line )
{
	stringstream ss ;

	ss << "#!/bin/bash\n" ;
	ss << "{}\n" ;

	ColorAssignment colorAssignment ;

	EXPECT_NO_THROW( ss >> colorAssignment ) ;
}



TEST( ColorAssignment, comment_without_space )
{
	stringstream ss ;

	ss << "#!/bin/bash\n" ;
	ss << "{\n" ;
	ss << "#node_type \n" ;
	ss << "}\n" ;

	ColorAssignment colorAssignment ;

	EXPECT_ANY_THROW( ss >> colorAssignment ) ;
}

