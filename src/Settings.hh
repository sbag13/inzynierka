#ifndef SETTINGS_HH
#define SETTINGS_HH



namespace microflow
{



/*
		Getters
*/


// TODO: define macros for getters and setters !!!



inline
double Settings::
getCharacteristicLengthLB() const
{
	return characteristicLengthLB_ ;
}



inline
double Settings::
getCharacteristicLengthPhysical() const
{
	return characteristicLengthPhysical_ ;
}



inline
double Settings::
getCharacteristicVelocityLB() const
{
	return characteristicVelocityLB_ ;
}



inline
double Settings:: 
getCharacteristicVelocityPhysical() const
{
	return characteristicVelocityPhysical_ ;
}



inline
double Settings::
getKinematicViscosityPhysical() const
{
	return kinematicViscosityPhysical_ ;
}



inline
double Settings::
getKinematicViscosityLB() const
{
	return kinematicViscosityLB_ ;
}



inline
double Settings::
getInitialVolumetricMassDensityLB() const
{
	return initialVolumetricMassDensityLB_ ;
}



inline
double Settings::
getInitialVolumetricMassDensityPhysical() const
{
	return initialVolumetricMassDensityPhysical_ ;
}



inline
double Settings::
getInitialVelocityLB( Axis axis ) const
{
	return initialVelocityLB_[ static_cast<unsigned>(axis) ] ;
}



inline
double Settings::
getLatticeSpacingPhysical() const
{
	return latticeSpacingPhysical_ ;
}



inline
double Settings::
getLatticeTimeStepPhysical() const
{
	return latticeTimeStepPhysical_ ;
}



inline
double Settings::
getTau() const
{
	return tau_ ;
}



inline
double Settings::
getReynoldsNumber() const
{
	return reynoldsNumber_ ;
}



inline
std::string Settings::
getSimulationDirectoryPath() const
{
	return simulationDirectoryPath_ ;
}



inline
std::string Settings::
getSettingsDirectoryPath() const
{
	return getSimulationDirectoryPath() + "/params/" ;
}



inline
std::string Settings::
getGeometryDirectoryPath() const
{
	return getSimulationDirectoryPath() + "/geometry/" ;
}



inline
std::string Settings::
getCheckpointDirectoryPath() const
{
	return getSimulationDirectoryPath() + "/checkpoint/" ;
}



inline
std::string Settings::
getOutputDirectoryPath() const
{
	return getSimulationDirectoryPath() + "/output/" ;
}



inline 
std::string Settings::
getPixelColorDefinitionsFilePath() const
{
	return getGeometryDirectoryPath() + "color_assignment" ;
}



inline
std::string Settings::
getGeometryPngImagePath() const
{
	return getGeometryDirectoryPath() + "geometry.png" ;
}



inline
std::string Settings::
getGeometryVtiImagePath() const
{
	return getGeometryDirectoryPath() + "geometry.vti" ;
}



inline
std::string Settings::
getInitialGeometryModificatorPath() const
{
	return getGeometryDirectoryPath() + "initialGeometryModificator.rb" ;
}



inline
std::string Settings::
getFinalGeometryModificatorPath() const
{
	return getGeometryDirectoryPath() + "finalGeometryModificator.rb" ;
}



inline
std::string Settings::
getLatticeArrangementName() const 
{
	return latticeArrangementName_ ;
}



inline
std::string Settings::
getDataTypeName() const
{
	return dataTypeName_ ;
}



inline
std::string Settings::
getFluidModelName() const
{
	return fluidModelName_ ;
}



inline
std::string Settings::
getCollisionModelName() const
{
	return collisionModelName_ ;
}



inline
std::string Settings::
getComputationalEngineName() const
{
	return computationalEngineName_ ;
}



inline
unsigned Settings::
getZExpandDepth() const
{
	return zExpandDepth_ ;
}



inline
unsigned Settings::
getNumberOfStepsBetweenVtkSaves() const
{
	return numberOfStepsBetweenVtkSaves_ ;
}



inline
unsigned Settings::
getMaxNumberOfVtkFiles() const
{
	return maxNumberOfVtkFiles_ ;
}



inline
bool Settings::
shouldSaveVtkInThisStep( unsigned stepNumber ) const
{
	if ( 0 == getMaxNumberOfVtkFiles() ) return false ;
	if ( 0 == getNumberOfStepsBetweenVtkSaves() ) return false ;
	if ( 0 == (stepNumber % getNumberOfStepsBetweenVtkSaves() ) ) return true ;
	return false ;		
}



inline
unsigned Settings::
getNumberOfStepsBetweenCheckpointSaves() const
{
	return numberOfStepsBetweenCheckpointSaves_ ;
}



inline
unsigned Settings::
getMaxNumberOfCheckpoints() const
{
	return maxNumberOfCheckpoints_ ;
}



inline
bool Settings::
shouldSaveCheckpointInThisStep( unsigned stepNumber ) const
{
	if ( 0 == getMaxNumberOfCheckpoints() ) return false ;
	if ( 0 == getNumberOfStepsBetweenCheckpointSaves() ) return false ;
	if ( 0 == (stepNumber % getNumberOfStepsBetweenCheckpointSaves() ) ) return true ;
	return false ;		
}



inline
bool Settings::
shouldSaveVelocityLB() const
{
	return shouldSaveVelocityLB_ ;
}



inline
bool Settings::
shouldSaveVelocityPhysical() const
{
	return shouldSaveVelocityPhysical_ ;
}



inline
bool Settings::
shouldSaveVolumetricMassDensityLB() const
{
	return shouldSaveVolumetricMassDensityLB_ ;
}



inline
bool Settings::
shouldSavePressurePhysical() const
{
	return shouldSavePressurePhysical_ ;
}



inline
bool Settings::
shouldSaveNodes() const
{
	return shouldSaveNodes_ ;
}



inline
bool Settings::
shouldSaveMassFlowFractions() const
{
	return shouldSaveMassFlowFractions_ ;
}



inline
unsigned Settings::
getNumberOfStepsBetweenErrorComputation() const
{
	return numberOfStepsBetweenErrorComputation_ ;
}



inline
bool Settings::
shouldComputeErrorInThisStep( unsigned stepNumber ) const
{
	if ( 0 == getNumberOfStepsBetweenErrorComputation() ) return false ;
	if ( 0 == (stepNumber % getNumberOfStepsBetweenErrorComputation() ) ) return true ;
	return false ;		
}



inline
double Settings::
getRequiredVelocityRelativeError() const
{
	return requiredVelocityRelativeError_ ;
}




inline NodeType Settings::getDefaultWallNode() const
{
	return defaultWallNode_ ;
}



inline NodeType Settings::getDefaultExternalCornerNode() const
{
	return defaultExternalCornerNode_ ;
}



inline NodeType Settings::getDefaultInternalCornerNode() const
{
	return defaultInternalCornerNode_ ;
}



inline NodeType Settings::getDefaultExternalEdgeNode() const
{
	return defaultExternalEdgeNode_ ;
}



inline NodeType Settings::getDefaultInternalEdgeNode() const
{
	return defaultInternalEdgeNode_ ;
}



inline NodeType Settings::getDefaultNotIdentifiedNode() const
{
	return defaultNotIdentifiedNode_ ;
}



inline NodeType Settings::getDefaultExternalEdgePressureNode() const
{
	return defaultExternalEdgePressureNode_ ;
}



inline NodeType Settings::getDefaultExternalCornerPressureNode() const
{
	return defaultExternalCornerPressureNode_ ;
}



inline NodeType Settings::getDefaultEdgeToPerpendicularWallNode() const
{
	return defaultEdgeToPerpendicularWallNode_ ;
}



inline const ModificationRhoU & Settings::getModificationRhoU() const
{
	return modificationRhoU_ ;
}



inline
Settings::DefaultValue Settings::
getVtkDefaultRhoForBB2Nodes() const
{
	if ("mean" == vtkDefaultRhoForBB2Nodes_)
	{
		return DefaultValue::MEAN ;
	}

	return DefaultValue::NOT_A_NUMBER ;
}



inline
UniversalCoordinates<double> Settings::
getGeometryOrigin() const
{
	return geometryOrigin_ ;
}



/*
		Setters
*/



inline
void Settings::
setCharacteristicLengthPhysical( double lengthPhys )
{
	 characteristicLengthPhysical_ = lengthPhys ;
}



inline
void Settings::
setCharacteristicVelocityPhysical( double velocityPhys )
{
	characteristicVelocityPhysical_ = velocityPhys ;
}



inline
void Settings::
setKinematicViscosityPhysical( double nuPhys )
{
	 kinematicViscosityPhysical_ = nuPhys ;
}



inline
void Settings::
setCharacteristicLengthLB( double lengthLB )
{
		characteristicLengthLB_ = lengthLB ;
}



inline
void Settings::
setInitialVolumetricMassDensityLB( double rhoLB )
{
	 initialVolumetricMassDensityLB_ = rhoLB ;
}



inline
void Settings::
setInitialVolumetricMassDensityPhysical( double rhoPhys )
{
	 initialVolumetricMassDensityPhysical_ = rhoPhys ;
}



inline
void Settings::
setInitialVelocityLB( Axis axis, double uLB ) 
{
	 initialVelocityLB_[ static_cast<unsigned>(axis) ] = uLB ;
}



inline
void Settings::
setTau( double tau )
{
	tau_ = tau ;
}



inline
void Settings::
setGeometryOrigin (UniversalCoordinates<double> origin)
{
	geometryOrigin_ = origin ;
}



/*
		Computations
*/



// TODO: call automatically in getters, when needed (maybe additional flag set in setters ?).
inline
void Settings::
recalculateCoefficients()
{
		reynoldsNumber_ = getCharacteristicVelocityPhysical() * 
											getCharacteristicLengthPhysical()   / 
											getKinematicViscosityPhysical() ;

		latticeSpacingPhysical_ = getCharacteristicLengthPhysical() / 
															getCharacteristicLengthLB() ; 

		double Cu  = 3.0 / (getTau() - 0.5) * 
								(getKinematicViscosityPhysical() / getLatticeSpacingPhysical()) ;

		characteristicVelocityLB_ = getCharacteristicVelocityPhysical() / Cu ;

		latticeTimeStepPhysical_ = getCharacteristicLengthPhysical() / 
															 getCharacteristicVelocityPhysical() * 
															 getCharacteristicVelocityLB() / getCharacteristicLengthLB() ; 

		kinematicViscosityLB_ = getCharacteristicVelocityLB() * getCharacteristicLengthLB() / 
														getReynoldsNumber() ; 
}



inline
double Settings::
transformVelocityLBToPhysical( double velocityLB ) const
{
	return ( velocityLB * getCharacteristicLengthLB() * getKinematicViscosityPhysical() ) / 
	         ( getKinematicViscosityLB() * getCharacteristicLengthPhysical() ) ;
}



inline
double Settings::
transformVelocityPhysicalToLB( double velocityPhysical ) const
{
	return (velocityPhysical * getCharacteristicLengthPhysical() * getKinematicViscosityLB()) /
				 (getKinematicViscosityPhysical() * getCharacteristicLengthLB()) ;
}



inline
double Settings::
transformVolumetricMassDensityLBToPressurePhysical( double volumetricMassDensityLB ) const
{
	return 1.0/3.0 * (volumetricMassDensityLB - getInitialVolumetricMassDensityLB()) * 
					getInitialVolumetricMassDensityPhysical() * 
					( (getLatticeSpacingPhysical() * getLatticeSpacingPhysical() ) / 
					  (getLatticeTimeStepPhysical() * getLatticeTimeStepPhysical() ) ) ;
}



inline
double Settings::
transformPressurePhysicalToVolumetricMassDensityLB( double pressurePhysical ) const
{
	return getInitialVolumetricMassDensityLB() + 
				 3.0 * ( pressurePhysical / getInitialVolumetricMassDensityPhysical() ) *
				 ( (getLatticeTimeStepPhysical() * getLatticeTimeStepPhysical()) /
				 	 (getLatticeSpacingPhysical() * getLatticeSpacingPhysical()) ) ;
}



}
#endif
