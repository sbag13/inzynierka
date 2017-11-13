#include "gtest/gtest.h"
#include "ClassificatorBoundaryAtLocation.hpp"
#include "Settings.hpp"

#include <memory>



using namespace std ;
using namespace microflow ;



TEST (ClassificatorBoundaryAtLocation, createFromNonExistentDirectory)
{
	EXPECT_ANY_THROW (ClassificatorBoundaryAtLocation ("nonExistingDirectory")) ;
}



TEST (ClassificatorBoundaryAtLocation, createFromCross200)
{
	unique_ptr <Settings> settings ;

	EXPECT_NO_THROW 
	(
		settings.reset (new Settings ("test_data/cases/cross/200x200x200"))
	) ;

	EXPECT_NO_THROW 
	(
		ClassificatorBoundaryAtLocation (settings->getGeometryDirectoryPath()) ;
	) ;
}



TEST (ClassificatorBoundaryAtLocation, getBoundaryDefinitions_Cross200)
{
	unique_ptr <Settings> settings ;
	unique_ptr <ClassificatorBoundaryAtLocation> classificator ;

	EXPECT_NO_THROW 
	(
		settings.reset (new Settings ("test_data/cases/cross/200x200x200")) ;
		classificator.reset 
			(new ClassificatorBoundaryAtLocation (settings->getGeometryDirectoryPath())) ;
	) ;

	auto boundaryDefinitions = classificator->getBoundaryDefinitions() ;

	EXPECT_EQ (0.0   , boundaryDefinitions.getBoundaryPressure  (0)) ;
	EXPECT_EQ (0.0001, boundaryDefinitions.getBoundaryVelocityX (0)) ;
	EXPECT_EQ (0.0   , boundaryDefinitions.getBoundaryVelocityY (0)) ;
	EXPECT_EQ (0.0   , boundaryDefinitions.getBoundaryVelocityZ (0)) ;
}
