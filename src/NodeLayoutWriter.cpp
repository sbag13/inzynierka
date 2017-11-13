#include "NodeLayoutWriter.hpp"

#include "gzstream.h"



namespace microflow
{



void NodeLayoutWriter::
saveToVolStream( const NodeLayout & nodeLayout, std::ostream & stream )
{
	Size size = nodeLayout.getSize() ;

	// FIXME: ugly hack, rebuild ?
	const NodeType & firstNode = nodeLayout.getNodeType(0, 0, 0) ;

	// Based on reverse engineering of QVox 2.8Beta-1 .vol output
	stream << "X: " << size.getWidth () << "\n" ;
	stream << "Y: " << size.getHeight() << "\n" ;
	stream << "Z: " << size.getDepth () << "\n" ;
	stream << "Version: 2\n" ;
	stream << "Voxel-Size: " << sizeof( firstNode ) << "\n" ;
	stream << "Alpha-Color: 0\n" ;
	stream << "Int-Endian: 0123\n" ;
	stream << "Voxel-Endian: 0\n" ;
	stream << "Res-X: 1.000000\n" ;
	stream << "Res-Y: 1.000000\n" ;
	stream << "Res-Z: 1.000000\n" ;
	stream << ".\n" ;

	//FIXME: ugly hack, rebuild ?
	// WARNING - works only for current nodeTypes_ layout - z,y,x.
	stream.write( (const char *)&(firstNode), size.computeVolume() * sizeof(firstNode) ) ;
}



void NodeLayoutWriter::
saveToVolFile( const NodeLayout & nodeLayout, std::string fileName )
{
	std::string gzFileName = fileName + ".vol.gz" ;
	ogzstream file( gzFileName.c_str() ) ;	

	saveToVolStream( nodeLayout, file ) ;
}



//FIXME: unfinished, handles only solid and non-solid nodes.
//FIXME: untested.
void NodeLayoutWriter::
saveToPngFile( const NodeLayout & nodeLayout, std::string fileName )
{
	Size size = nodeLayout.getSize() ;

	Image image( size.getWidth(), size.getHeight() ) ;

	for (unsigned x=0 ; x < size.getWidth() ; x++)
		for (unsigned y=0 ; y < size.getHeight() ; y++)
		{
			// z=1, because top and bottom are modified
			NodeType node = nodeLayout.getNodeType( x,y,1 ) ;
			Pixel p(255,255,255) ;

			if ( isFluid( node.getBaseType() ) )
			{
				p = Pixel(0,0,0) ;
			}
			image.setPixel( x,y, p ) ;
		}

	image.write( fileName ) ;
}



}
