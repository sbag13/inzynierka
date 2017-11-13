#ifndef BOUNDARY_DESCRIPTION_HH
#define BOUNDARY_DESCRIPTION_HH



#include <cmath>



namespace microflow
{



inline
BoundaryDescription::
BoundaryDescription()
{
	isCharacteristicLengthMarker_ = false ;

	nodeBaseType_ = NodeBaseType::SOLID ; // TODO: set to UNKNOWN ?

	pressure_    = NAN ;
	velocity_[0] = NAN ;
	velocity_[1] = NAN ;
	velocity_[2] = NAN ;
}



inline
bool BoundaryDescription::
isSolid() const
{
	return microflow::isSolid( nodeBaseType_ ) ;
}



inline
bool BoundaryDescription::
isFluid() const
{
	return microflow::isFluid( nodeBaseType_ ) ;
}



inline
bool BoundaryDescription::
isBoundary() const
{
	return microflow::isBoundary( nodeBaseType_ ) ;
}



inline
bool BoundaryDescription::
isCharacteristicLengthMarker() const
{
	return isCharacteristicLengthMarker_ ;
}



inline
const double (& BoundaryDescription::getVelocity() const) [3]
{
	return velocity_ ;
}



inline
double BoundaryDescription::
getPressure() const
{
	return pressure_ ;
}



inline
NodeBaseType BoundaryDescription::
getNodeBaseType() const
{
	return nodeBaseType_ ;
}



}



#endif
