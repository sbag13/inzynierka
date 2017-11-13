#ifndef COLORED_PIXEL_CLASSIFICATOR_HPP
#define COLORED_PIXEL_CLASSIFICATOR_HPP



#include <png.hpp>
#include <vector>

#include "NodeType.hpp"
#include "ColorAssignment.hpp"
#include "BoundaryDefinitions.hpp"



namespace microflow
{



class ColoredPixelClassificator
{
	public:
		ColoredPixelClassificator( std::string pathToColorAssignmentFile ) ;
		~ColoredPixelClassificator() ;

		NodeType createNode( const png::rgb_pixel & pixel ) const ;

		BoundaryDefinitions getBoundaryDefinitions() const ;

	private:
		int findColorAssignment( const png::rgb_pixel & color,
														 const std::vector< ColorAssignment > & colorAssignment ) const ;

		std::vector< ColorAssignment > solidFluidAssignments_ ;
		std::vector< ColorAssignment > boundaryAssignments_ ;
} ;



}



#endif
