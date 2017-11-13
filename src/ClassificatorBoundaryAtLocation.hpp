#ifndef CLASSIFICATOR_BOUNDARY_AT_LOCATION_HPP
#define CLASSIFICATOR_BOUNDARY_AT_LOCATION_HPP



#include <vector>

#include "BoundaryAtLocations.hpp"



namespace microflow
{



class ClassificatorBoundaryAtLocation
{
	public:

		ClassificatorBoundaryAtLocation (std::string pathToGeometryDirectory) ;

		BoundaryDefinitions getBoundaryDefinitions() const ;
		void setBoundaryNodes (NodeLayout & nodeLayout) ;


	private:
		
		std::vector <BoundaryAtLocations> boundaryAtLocations_ ;
} ;



}



#endif
