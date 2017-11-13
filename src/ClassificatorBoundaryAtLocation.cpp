#include <fstream>

#include "ClassificatorBoundaryAtLocation.hpp"
#include "Axis.hpp"



using namespace std ;
using namespace microflow ;



ClassificatorBoundaryAtLocation::
ClassificatorBoundaryAtLocation (string pathToGeometryDirectory)
{
	ifstream boundaryLocationsFile ;
	string boundaryLocationsFilePath = pathToGeometryDirectory + 
																		 "/boundary_locations" ;

	boundaryLocationsFile.open (boundaryLocationsFilePath) ;
	if (boundaryLocationsFile.fail())
	{
		THROW ("Can not open file \"" + boundaryLocationsFilePath + "\"") ;
	}

	logger << "Loading boundary node description from \""
				 << boundaryLocationsFilePath << "\"\n" ;


	while (true)
	{
		BoundaryAtLocations boundary ;

		try
		{
			boundary.read (boundaryLocationsFile) ;
		}
		catch (...)
		{
			THROW ("Syntax error, exiting") ;
		}

		if (boundaryLocationsFile.eof())
		{
			break ;
		}

		boundaryAtLocations_.push_back (boundary) ;
	}

	logger << "OK, loaded " << boundaryAtLocations_.size()
				 << " boundary descriptions.\n" ;


	for (auto & b : boundaryAtLocations_)
	{
		b.loadLocationFiles (pathToGeometryDirectory) ;
	}
}



BoundaryDefinitions ClassificatorBoundaryAtLocation::
getBoundaryDefinitions() const
{
	BoundaryDefinitions boundaryDefinitions ;

	for (auto boundary : boundaryAtLocations_)
	{
		boundaryDefinitions.addBoundaryDefinition 
			(
				boundary.getVelocity() [X],
				boundary.getVelocity() [Y],
				boundary.getVelocity() [Z],
				boundary.getPressure()
			) ;
	}

	return boundaryDefinitions ;
}



void ClassificatorBoundaryAtLocation::
setBoundaryNodes (NodeLayout & nodeLayout)
{
	logger << "Setting boundary nodes:\n" ;

	for (unsigned char boundaryDefinitionIndex = 0 ; 
			 boundaryDefinitionIndex < boundaryAtLocations_.size() ;
			 boundaryDefinitionIndex ++)
	{
#define boundary (boundaryAtLocations_ [boundaryDefinitionIndex])

		NodeType boundaryNode (boundary.getNodeBaseType()) ;
		boundaryNode.setBoundaryDefinitionIndex (boundaryDefinitionIndex) ;

		auto & nodeLocations = boundary.getNodeLocations() ;

		logger << "Boundary condition " << (int)(boundaryDefinitionIndex)
					 << ": " << boundaryNode 
					 << ", " << nodeLocations.size() << " nodes"
					 << "\n" ;

		for (auto nodeCoordinate : nodeLocations)
		{
			nodeLayout.setNodeType (nodeCoordinate, boundaryNode) ;
		}
#undef boundary
	}

	logger << "Boundary nodes set.\n" ;
}
