#include <algorithm>
#include <iostream>

#include "NodeLayout.hpp"
#include "Image.hpp"
#include "Exceptions.hpp"



using namespace std ;



namespace microflow
{



constexpr const enum NodeBaseType NodeLayout::BOUNDARY_MARKER ;



NodeLayout::
NodeLayout( const ColoredPixelClassificator & coloredPixelClassificator,
						const Image & image,
						size_t depth)
:	boundaryDefinitions_( coloredPixelClassificator.getBoundaryDefinitions() )
{
	Size size( image.get_width(), image.get_height(), depth ) ;
	nodeTypes_.resize( size, NodeBaseType::SOLID ) ;

	for (size_t z=0 ; z < size.getDepth() ; z++)
		for (size_t y=0 ; y < size.getHeight() ; y++)
			for (size_t x=0 ; x < size.getWidth() ; x++ )
			{
				setNodeType( x, y, z,
						coloredPixelClassificator.createNode( image.getPixel( x, y ) )
				) ;
			}
}



NodeLayout::
NodeLayout( const Size & size )
{
	nodeTypes_.resize( size, NodeBaseType::SOLID ) ;
}



void NodeLayout::
resizeWithContent( const Size & newSize )
{
	nodeTypes_.resizeWithContent( newSize, NodeBaseType::SOLID ) ;
}



void NodeLayout::
restoreBoundaryNodes(const ColoredPixelClassificator & coloredPixelClassificator,
										 const Image & image )
{
	const Size size = getSize() ;
	if ( image.get_width() != size.getWidth()  ||  
			 image.get_height() != size.getHeight()  )
	{
		THROW ("Image size differs from NodeLayout size.") ;
	}
	
	//FIXME: remember about top and bottom covers
	for (size_t z=1 ; z < size.getDepth()-1 ; z++)
		for (size_t y=0 ; y < size.getHeight() ; y++)
			for (size_t x=0 ; x < size.getWidth() ; x++ )
			{
				NodeType imageNode = coloredPixelClassificator.
					createNode( image.getPixel( x, y ) ) ;

				if ( imageNode.isBoundary() ) 
				{
					setNodeType( x, y, z, imageNode ) ;
				}
			}
}



}
