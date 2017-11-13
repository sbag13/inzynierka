#include "gtest/gtest.h"
#include "BoundaryAtLocations.hpp"

#include <sstream>



using namespace microflow ;
using namespace std ;



TEST (BoundaryAtLocations, emptyConstructor)
{
	BoundaryAtLocations boundaryAtLocations ;

	EXPECT_EQ (0u, boundaryAtLocations.getFileNames().size()) ;
}



TEST (BoundaryAtLocations, readGeneral)
{
	stringstream ss ;

	ss << "# \n" ;
	ss << "{\n" ;
	ss << "# node_type \n" ;
	ss << "# some comment \n" ;
	ss << "# \n" ;
	ss << "node_type = solid\n" ;
	ss << "velocity = (1.0, 2, 3.5)\n" ;
	ss << "pressure = 5.5\n" ;
	ss << "file   =  file1.csv \n" ;
	ss << "file   =  file2.csv \n" ;
	ss << "file   =  file3.csv \n" ;
	ss << "}\n" ;
	
	BoundaryAtLocations boundaryAtLocations ;

	EXPECT_NO_THROW (boundaryAtLocations.read (ss)) ;

	auto const & fileNames = boundaryAtLocations.getFileNames() ;
	EXPECT_EQ (3u, fileNames.size()) ;
	EXPECT_EQ ("file1.csv", fileNames[0]) ;
	EXPECT_EQ ("file2.csv", fileNames[1]) ;
	EXPECT_EQ ("file3.csv", fileNames[2]) ;

	EXPECT_TRUE (boundaryAtLocations.isSolid()) ;
	EXPECT_EQ (NodeBaseType::SOLID, boundaryAtLocations.getNodeBaseType()) ;

	EXPECT_EQ( 1.0, boundaryAtLocations.getVelocity()[0] ) ;
	EXPECT_EQ( 2.0, boundaryAtLocations.getVelocity()[1] ) ;
	EXPECT_EQ( 3.5, boundaryAtLocations.getVelocity()[2] ) ;

	EXPECT_EQ( 5.5, boundaryAtLocations.getPressure() ) ;
}



TEST (BoundaryAtLocations, exceptionReadColor)
{
	stringstream ss ;

	ss << "# \n" ;
	ss << "{\n" ;
	ss << "# node_type \n" ;
	ss << "# some comment \n" ;
	ss << "# \n" ;
	ss << "node_type = solid\n" ;
	ss << "velocity = (1.0, 2, 3.5)\n" ;
	ss << "color  =  (128,129,130) \n" ;
	ss << "}\n" ;

	BoundaryAtLocations boundaryAtLocations ;

	EXPECT_ANY_THROW (boundaryAtLocations.read (ss)) ;
}
