#ifndef BOUNDARY_DEFINITIONS_HPP
#define BOUNDARY_DEFINITIONS_HPP



namespace microflow
{



#include <vector>



class BoundaryDefinitions
{
	public:

		void addBoundaryDefinition( double velocityX, double velocityY, double velocityZ, 
																double pressure ) ;

		double getBoundaryPressure( unsigned short boundaryDefinitionIndex ) const ;
		double getBoundaryVelocityX( unsigned short boundaryDefinitionIndex ) const ;
		double getBoundaryVelocityY( unsigned short boundaryDefinitionIndex ) const ;
		double getBoundaryVelocityZ( unsigned short boundaryDefinitionIndex ) const ;

	private:
		std::vector<double> pressure_ ;
		std::vector<double> velocityX_ ;
		std::vector<double> velocityY_ ;
		std::vector<double> velocityZ_ ;
} ;



}



#include "BoundaryDefinitions.hh"



#endif
