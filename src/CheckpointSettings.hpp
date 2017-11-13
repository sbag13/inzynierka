#ifndef CHECKPOINT_SETTINGS_HPP
#define CHECKPOINT_SETTINGS_HPP



#include "Exceptions.hpp"
#include "Coordinates.hpp"



namespace microflow
{



class CheckpointSettings
{
	public:
		bool shouldSaveVelocityLB             () const { return true ; }
		bool shouldSaveVolumetricMassDensityLB() const { return true ; }
		bool shouldSaveMassFlowFractions      () const { return true ; }
		bool shouldSaveNodes                  () const { return true ; }

		bool shouldSaveVelocityPhysical() const { return false ; }
		bool shouldSavePressurePhysical() const { return false ; }

		UniversalCoordinates<double> getGeometryOrigin() const
		{
			return UniversalCoordinates<double> (0,0,0) ;
		}
		double getLatticeSpacingPhysical() const { return 1.0 ; }

		double transformVelocityLBToPhysical (double) const
		{ 
			THROW("UNIMPLEMENTED") ;
			return NAN ;
		}
		double transformVolumetricMassDensityLBToPressurePhysical (double) const
		{
			THROW("UNIMPLEMENTED") ;
			return NAN ;
		}
} ;



}



#endif
