#include "gtest/gtest.h"
#include "ColoredPixelClassificator.hpp"

#include <string>

#include "ColoredPixelClassificatorTest.hpp"



using namespace microflow ;



TEST( ColoredPixelClassificator, create_from_non_existing_file )
{
	EXPECT_ANY_THROW( ColoredPixelClassificator( "nonExisstingFile" ) ) ;
}



std::string colorAssignmentFilePath("./test_data/geometries/color_assignment") ;

ColoredPixelClassificator getDefaultPixelClassificator()
{
	return ColoredPixelClassificator( colorAssignmentFilePath ) ;
}



TEST( ColoredPixelClassificator, input_file_correct )
{
	EXPECT_NO_THROW( 
		ColoredPixelClassificator coloredPixelClassificator( colorAssignmentFilePath ) 
	) ;
}



TEST( ColoredPixelClassificator, solid_nodes )
{
	ColoredPixelClassificator coloredPixelClassificator = getDefaultPixelClassificator() ;

	EXPECT_TRUE( coloredPixelClassificator.createNode( png::rgb_pixel(255,255,255) ).isSolid() ) ;
	EXPECT_TRUE( coloredPixelClassificator.createNode( png::rgb_pixel(200,200,200) ).isSolid() ) ;

	EXPECT_TRUE( coloredPixelClassificator.createNode( png::rgb_pixel(201,202,203) ).isSolid() ) ;
}



TEST( ColoredPixelClassificator, fluid_nodes )
{
	ColoredPixelClassificator coloredPixelClassificator = getDefaultPixelClassificator() ;

	EXPECT_FALSE( coloredPixelClassificator.createNode( png::rgb_pixel(0,0,0) ).isSolid() ) ;
	EXPECT_FALSE( coloredPixelClassificator.createNode( png::rgb_pixel(50,50,50) ).isSolid() ) ;
	EXPECT_TRUE ( coloredPixelClassificator.createNode( png::rgb_pixel(0,0,0) ).isFluid() ) ;
	EXPECT_TRUE ( coloredPixelClassificator.createNode( png::rgb_pixel(50,50,50) ).isFluid() ) ;
}



TEST( ColoredPixelClassificator, velocity )
{
	ColoredPixelClassificator coloredPixelClassificator = getDefaultPixelClassificator() ;

	NodeType node( coloredPixelClassificator.createNode( png::rgb_pixel(1,1,1) ) )  ;

	EXPECT_EQ( NodeBaseType::VELOCITY, node.getBaseType() ) ;
	EXPECT_EQ( 0, node.getBoundaryDefinitionIndex() ) ;
}



TEST( ColoredPixelClassificator, velocity_0 )
{
	ColoredPixelClassificator coloredPixelClassificator = getDefaultPixelClassificator() ;

	NodeType node( coloredPixelClassificator.createNode( png::rgb_pixel(2,2,2) ) )  ;

	EXPECT_EQ( NodeBaseType::VELOCITY_0, node.getBaseType() ) ;
	EXPECT_EQ( 1, node.getBoundaryDefinitionIndex() ) ;
}



TEST( ColoredPixelClassificator, bounce_back_2 )
{
	ColoredPixelClassificator coloredPixelClassificator = getDefaultPixelClassificator() ;

	NodeType node( coloredPixelClassificator.createNode( png::rgb_pixel(3,3,3) ) )  ;

	EXPECT_EQ( NodeBaseType::BOUNCE_BACK_2, node.getBaseType() ) ;
	EXPECT_EQ( 2, node.getBoundaryDefinitionIndex() ) ;
}



TEST( ColoredPixelClassificator, pressure )
{
	ColoredPixelClassificator coloredPixelClassificator = getDefaultPixelClassificator() ;

	NodeType node( coloredPixelClassificator.createNode( png::rgb_pixel(4,4,4) ) )  ;

	EXPECT_EQ( NodeBaseType::PRESSURE, node.getBaseType() ) ;
	EXPECT_EQ( 3, node.getBoundaryDefinitionIndex() ) ;
}

