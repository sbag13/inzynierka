#ifndef BOUNDARY_DEFINITIONS_HH
#define BOUNDARY_DEFINITIONS_HH



namespace microflow
{



inline
void BoundaryDefinitions::
addBoundaryDefinition( double velocityX, double velocityY, double velocityZ, 
											 double pressure )
{
	pressure_.push_back( pressure ) ;
	velocityX_.push_back( velocityX ) ;
	velocityY_.push_back( velocityY ) ;
	velocityZ_.push_back( velocityZ ) ;
}



inline
double BoundaryDefinitions::
getBoundaryPressure( unsigned short boundaryDefinitionIndex ) const
{
	if ( boundaryDefinitionIndex < pressure_.size() )
	{
		return pressure_[ boundaryDefinitionIndex ] ;
	}
	else
	{
		//TODO: I am not sure, if this is a good way to define default values...
		return 0 ; //NAN ;
	}
}



inline
double BoundaryDefinitions::
getBoundaryVelocityX( unsigned short boundaryDefinitionIndex ) const
{
	if ( boundaryDefinitionIndex < velocityX_.size() )
	{
		return velocityX_[ boundaryDefinitionIndex ] ;
	}
	else
	{
		return 0 ; //NAN ;
	}
}



inline
double BoundaryDefinitions::
getBoundaryVelocityY( unsigned short boundaryDefinitionIndex ) const
{
	if ( boundaryDefinitionIndex < velocityY_.size() )
	{
		return velocityY_[ boundaryDefinitionIndex ] ;
	}
	else
	{
		return 0 ; //NAN ;
	}
}



inline
double BoundaryDefinitions::
getBoundaryVelocityZ( unsigned short boundaryDefinitionIndex ) const
{
	if ( boundaryDefinitionIndex < velocityZ_.size() )
	{
		return velocityZ_[ boundaryDefinitionIndex ] ;
	}
	else
	{
		return 0 ; //NAN ;
	}
}



}



#endif
