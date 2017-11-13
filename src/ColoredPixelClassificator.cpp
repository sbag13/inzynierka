#include <fstream>


#include "Logger.hpp"
#include "ColoredPixelClassificator.hpp"
#include "Exceptions.hpp"
#include "Image.hpp"
#include "Axis.hpp"



using namespace std ;



namespace microflow
{



ColoredPixelClassificator::
ColoredPixelClassificator( std::string pathToColorAssignmentFile )
{
	ifstream file ;
  
	file.open( pathToColorAssignmentFile ) ;
	if ( file.fail() )
	{
		throw std::exception() ;
	}

	logger << "Loading color assignment from \"" ;
	logger << pathToColorAssignmentFile << "\"\n" ;

	while (true)
	{
		ColorAssignment colorAssignment ;

		try 
		{
			file >> colorAssignment ;
		}
		catch(...)
		{
			THROW ("Syntax error, exiting\n") ;
		}

		if ( file.eof() )
		{
			break ;
		}

		// TODO: if performance is low, consider sorting colorAssignments_ 
		//       by frequency of use (fluid	nodes first).
		if ( colorAssignment.isBoundary() )
		{
			boundaryAssignments_.push_back( colorAssignment ) ;
		} 
		else
		{
			solidFluidAssignments_.push_back( colorAssignment ) ;
		}
	}	
}



ColoredPixelClassificator::
~ColoredPixelClassificator()
{}



NodeType ColoredPixelClassificator::
createNode( const png::rgb_pixel & pixel ) const
{
	NodeType nodeType ;

	int index = findColorAssignment( pixel, solidFluidAssignments_ ) ;
	if ( 0 <= index )
	{
		ColorAssignment colorAssignment = solidFluidAssignments_[ index ] ;
		
		nodeType.setBaseType( colorAssignment.getNodeBaseType() ) ;
	}
	else
	{
		index = findColorAssignment( pixel, boundaryAssignments_ ) ;
		if ( 0 <= index )
		{
			ColorAssignment colorAssignment = boundaryAssignments_[ index ] ;

			nodeType.setBaseType( colorAssignment.getNodeBaseType() ) ;
			nodeType.setBoundaryDefinitionIndex( index ) ;
		}
		else
		{
			logger << "WARNING: found undefined color " ;
			logger << pixel << ", assuming solid\n" ;
		}
	}

	return nodeType ;
}



BoundaryDefinitions ColoredPixelClassificator::
getBoundaryDefinitions() const
{
	BoundaryDefinitions boundaryDefinitions ;

	for (auto it =  boundaryAssignments_.begin() ;
						it !=	boundaryAssignments_.end() ;
						it++)
	{
		boundaryDefinitions.addBoundaryDefinition( 
																								it->getVelocity()[X], 
																								it->getVelocity()[Y], 
																								it->getVelocity()[Z], 
																								it->getPressure() ) ;
	}

	return boundaryDefinitions ;
}



int ColoredPixelClassificator::
findColorAssignment( const png::rgb_pixel & color,
		                 const std::vector< ColorAssignment > & colorAssignments ) const
{
	for (size_t i=0 ; i < colorAssignments.size() ; i++)
	{
		if ( colorAssignments[ i ].colorEquals( color ) )
		{
			return i ;
		}
	}

	return -1 ;
}



}
