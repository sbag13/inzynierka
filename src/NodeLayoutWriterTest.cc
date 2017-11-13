#include "gtest/gtest.h"
#include <sstream>

#include "NodeLayoutWriter.hpp"

#include "ColoredPixelClassificatorTest.hpp"



using namespace microflow ;
using namespace std ;



NodeLayout createNodeLayout(unsigned depth)
{
	ColoredPixelClassificator coloredPixelClassificator = getDefaultPixelClassificator() ;

	Image image(16,16) ;
	image.fill( Pixel(255,255,255) ) ;
	
	image.setPixel(0, 0, Pixel(0,0,0)) ;
	image.setPixel(1, 1, Pixel(0,0,0)) ;
	image.setPixel(1, 10, Pixel(0,0,0)) ;

	return NodeLayout (coloredPixelClassificator, image, depth) ;
}



void checkVolFileHeader(istream & str, unsigned dimX, unsigned dimY, unsigned dimZ)
{
	string s ;
	unsigned a ;

	str >> s ; EXPECT_EQ( "X:", s) ;
	str >> a ; EXPECT_EQ( dimX, a) ;
	str >> s ; EXPECT_EQ( "Y:", s) ;
	str >> a ; EXPECT_EQ( dimY, a) ;
	str >> s ; EXPECT_EQ( "Z:", s) ;
	str >> a ; EXPECT_EQ( dimZ, a) ;

	str >> s ; EXPECT_EQ( "Version:", s) ;
	str >> a ; EXPECT_EQ( 2u, a) ;

	str >> s ; EXPECT_EQ( "Voxel-Size:", s) ;
	str >> a ; EXPECT_EQ( 2u, a) ;

	str >> s ; EXPECT_EQ( "Alpha-Color:", s) ;
	str >> a ; EXPECT_EQ( 0u, a) ;

	str >> s ; EXPECT_EQ( "Int-Endian:", s) ;
	str >> s ; EXPECT_EQ( "0123", s) ;

	str >> s ; EXPECT_EQ( "Voxel-Endian:", s) ;
	str >> a ; EXPECT_EQ( 0u, a) ;

	double d ;
	str >> s ; EXPECT_EQ( "Res-X:", s) ;
	str >> d ; EXPECT_EQ( 1.0, d) ;
	str >> s ; EXPECT_EQ( "Res-Y:", s) ;
	str >> d ; EXPECT_EQ( 1.0, d) ;
	str >> s ; EXPECT_EQ( "Res-Z:", s) ;
	str >> d ; EXPECT_EQ( 1.0, d) ;

	str >> s ; EXPECT_EQ(".", s) ;

	getline( str, s ) ; // eat white
	ASSERT_TRUE( str ) ;
}



void testNodeLayoutWriter( unsigned depth )
{
	NodeLayout nodeLayout = createNodeLayout( depth ) ;

	stringstream str ;

	NodeLayoutWriter().saveToVolStream( nodeLayout, str ) ;

	checkVolFileHeader(str, 16, 16, depth) ;

	for (unsigned int z=0 ; z < depth ; z++)
		for (unsigned int y=0 ; y < 16 ; y++)
			for (unsigned int x=0 ; x < 16 ; x++)
			{
				NodeType nodeType ;

				str.read( (char*)(&nodeType), sizeof(nodeType) ) ;

				ASSERT_TRUE( str ) ;

				if (
						(0 == x  &&  0 == y) ||
						(1 == x  &&  1 == y) ||
						(1 == x  && 10 == y)
					 )
				{
					ASSERT_EQ( nodeType, NodeBaseType::FLUID ) ;
				}
				else
				{
					ASSERT_EQ( nodeType, NodeBaseType::SOLID ) ;
				}
			}

	char c ; // check if all read
	str.read( &c, 1 ) ;
	ASSERT_FALSE( str ) ;
}



TEST( NodeLayoutWriter, volFile2D )
{
	testNodeLayoutWriter(1) ;
}



TEST( NodeLayoutWriter, volFile3D_2 )
{
	testNodeLayoutWriter(2) ;
}



TEST( NodeLayoutWriter, volFile3D_10 )
{
	testNodeLayoutWriter(10) ;
}



TEST( NodeLayoutWriter, volFile3D_16 )
{
	testNodeLayoutWriter(16) ;
}



TEST( NodeLayoutWriter, volFile3D_17 )
{
	testNodeLayoutWriter(17) ;
}
