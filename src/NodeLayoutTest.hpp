#ifndef NODE_LAYOUT_TEST_HPP
#define NODE_LAYOUT_TEST_HPP



#include "NodeLayout.hpp"



namespace microflow
{



NodeLayout createHomogenousNodeLayout( unsigned width, unsigned height, unsigned depth,
																			 Pixel defaultPixel) ;
NodeLayout createSolidNodeLayout( unsigned width, unsigned height, unsigned depth) ;
NodeLayout createFluidNodeLayout( unsigned width, unsigned height, unsigned depth) ;

Image createSolidLayoutImage( unsigned width, unsigned height ) ;



}



#endif
