#ifndef BOUNDARY_DESCRIPTION_HPP
#define BOUNDARY_DESCRIPTION_HPP



#include <istream>

#include "NodeBaseType.hpp"



namespace microflow
{



/*
	Base class for all boundary descritpions (colored pixel, location).
*/
class BoundaryDescription
{
	public:

		BoundaryDescription() ;
		virtual ~BoundaryDescription() {} ;

		bool isSolid                      () const ;
		bool isFluid                      () const ;
		bool isBoundary                   () const ;
		bool isCharacteristicLengthMarker () const ;

		const double (& getVelocity() const) [3] ;
		double getPressure() const ;
		NodeBaseType getNodeBaseType() const ;

		void read (std::istream & stream) ;


	protected:

		virtual bool readElement (const std::string & elementName, 
															std::istream & stream) ;

		bool isCharacteristicLengthMarker_ ;
		NodeBaseType nodeBaseType_ ;

		double velocity_[3] ;
		double pressure_    ;
} ;



}



#include "BoundaryDescription.hh"



#endif
