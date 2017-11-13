#include "gtest/gtest.h"
#include <cstdlib>

#include "NodeLayout.hpp"

#include "NodeLayoutTest.hpp"
#include "ColoredPixelClassificatorTest.hpp"



using namespace microflow ;




namespace microflow
{



NodeLayout createHomogenousNodeLayout( unsigned width, unsigned height, unsigned depth,
																			 Pixel defaultPixel)
{
	Image image(width,height) ;
	image.fill( defaultPixel ) ;

	return NodeLayout (getDefaultPixelClassificator(), image, depth) ;
}



NodeLayout createSolidNodeLayout( unsigned width, unsigned height, unsigned depth)
{
	return createHomogenousNodeLayout( width, height, depth, Pixel(255,255,255) ) ;
}



NodeLayout createFluidNodeLayout( unsigned width, unsigned height, unsigned depth)
{
	return createHomogenousNodeLayout( width, height, depth, Pixel(0,0,0) ) ;
}



Image createRandomLayoutImage( unsigned width, unsigned height )
{
	Image image(width,height) ;

	srand(1) ;
	for (unsigned y=0 ; y < height ; y++)
		for (unsigned x=0 ; x < width ; x++)
		{
			unsigned r = rand() % 5 ;
			image.setPixel(x, y, Pixel(r,r,r)) ;
		}
	
	return image ;
}



Image createSolidLayoutImage( unsigned width, unsigned height )
{
	Image image(width,height) ;
	image.fill( Pixel(255,255,255) ) ;
	return image ;
}



}



TEST( NodeLayout, createFromImage )
{
	Image image = createSolidLayoutImage(16, 16) ;
	
	image.setPixel(0, 0, Pixel(0,0,0)) ;
	image.setPixel(1, 1, Pixel(0,0,0)) ;
	image.setPixel(1, 10, Pixel(0,0,0)) ;

	NodeLayout nodeLayout(getDefaultPixelClassificator(), image, 10) ;

	EXPECT_FALSE( nodeLayout.getNodeType(0,0,0).isSolid() ) ;
	EXPECT_FALSE( nodeLayout.getNodeType(0,0,1).isSolid() ) ;
	EXPECT_FALSE( nodeLayout.getNodeType(0,0,9).isSolid() ) ;

	EXPECT_EQ( nodeLayout.getNodeType(0,0,0), NodeBaseType::FLUID ) ;

	for (unsigned z=0 ; z < 10 ; z++)
		for (unsigned y=0 ; y < 16 ; y++)
			for (unsigned x=0 ; x < 16 ; x++)
			{
				if ( 
						( 0 == x  && 0 == y )  || 
					  ( 1 == x  && 1 == y )  ||
					  ( 1 == x  && 10 == y )  
					 )
				{
					ASSERT_TRUE( nodeLayout.getNodeType(x, y, z).isFluid() ) ;
				}
				else
				{
					ASSERT_TRUE( nodeLayout.getNodeType(x,y,z).isSolid() ) ;
				}
			}
}


TEST( NodeLayout, getNodeType_solid )
{
	unsigned width = 10, height = 10, depth = 10 ;

	NodeLayout nodeLayout = createSolidNodeLayout(width, height, depth) ;

	for (unsigned z=0 ; z < depth ; z++)
		for (unsigned y=0 ; y < height ; y++)
			for (unsigned x=0 ; x < width ; x++)
			{
				ASSERT_EQ( nodeLayout.getNodeType(x,y,z), NodeBaseType::SOLID ) ;
			}
}



TEST( NodeLayout, getNodeType_fluid )
{
	unsigned width = 10, height = 10, depth = 10 ;

	NodeLayout nodeLayout = createFluidNodeLayout(width, height, depth) ;

	for (unsigned z=0 ; z < depth ; z++)
		for (unsigned y=0 ; y < height ; y++)
			for (unsigned x=0 ; x < width ; x++)
			{
				ASSERT_EQ( nodeLayout.getNodeType(x,y,z), NodeBaseType::FLUID ) ;
			}
}


void testRandomNodeLayout( unsigned oldWidth, unsigned oldHeight, unsigned oldDepth,
													 unsigned newWidth, unsigned newHeight, unsigned newDepth )
{
	Image image = createRandomLayoutImage( oldWidth, oldHeight ) ;
	ColoredPixelClassificator coloredPixelClassificator = getDefaultPixelClassificator() ;
	NodeLayout nodeLayout(coloredPixelClassificator, image, oldDepth) ;

	nodeLayout.resizeWithContent( Size(newWidth, newHeight, newDepth) ) ;

	for (unsigned y=0 ; y < newHeight ; y++)
		for (unsigned x=0 ; x < newWidth ; x++)
			for (unsigned z=0 ; z < newDepth ; z++)
			{
				NodeType node = nodeLayout.getNodeType(x, y, z) ;

				Coordinates c (x,y,z) ;
				NodeType node2 = nodeLayout.getNodeType( c ) ;

				ASSERT_EQ( node, node2 ) ;


				if ( x < oldWidth  && y < oldHeight  && z < oldDepth )
				{
					Pixel pixel = image.getPixel(x, y) ;
					NodeType expectedNode = coloredPixelClassificator.createNode( pixel ) ;

					ASSERT_EQ( node, expectedNode.getBaseType() ) ;
				}
				else
				{
					ASSERT_EQ( node, NodeBaseType::SOLID ) ;
				}
			}
}



TEST( NodeLayout, getNodeType_random )
{
	testRandomNodeLayout( 0, 0, 0,
												0, 0, 0 ) ;
	testRandomNodeLayout( 1, 1, 1,
												1, 1, 1 ) ;
	testRandomNodeLayout( 2, 2, 2,
												2, 2, 2 ) ;
	testRandomNodeLayout( 20, 20, 20,
												20, 20, 20 ) ;
}



TEST( NodeLayout, resizeWithContent )
{
	testRandomNodeLayout( 1, 1, 1,
												2, 2, 2 ) ;
	testRandomNodeLayout( 3, 5, 7,
												5, 7, 9 ) ;
	testRandomNodeLayout( 5, 7, 9,
												3, 5, 7) ;
	testRandomNodeLayout( 20, 20, 20,
												40, 40, 40 ) ;
	testRandomNodeLayout( 20, 30, 40,
												40, 40, 40 ) ;
	testRandomNodeLayout( 20, 30, 40,
												1,  1,  1 ) ;
	testRandomNodeLayout( 20, 30, 40,
												0,  0,  0 ) ;
	testRandomNodeLayout( 0,  0,  0,
												10, 10, 10 ) ;
}



TEST( NodeLayout, getNodeType_outside )
{
	unsigned width = 10, height = 10, depth = 10 ;

	NodeLayout nodeLayout = createSolidNodeLayout(width, height, depth) ;

	ASSERT_DEATH( nodeLayout.getNodeType(width, height, depth), "" ) ;
}

