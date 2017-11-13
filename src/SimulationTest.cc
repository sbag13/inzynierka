#include "gtest/gtest.h"

#include "Simulation.hpp"
#include "Settings.hpp"



using namespace microflow ;
using namespace std ;



TEST( Simulation, createFromDirectory )
{
	EXPECT_NO_THROW
	(	
		Simulation simulation("./test_data/cases/configuration_3") ;
	) ;
}



TEST( Simulation, createFrom_simulation_1 )
{
	EXPECT_NO_THROW
	(	
		Simulation simulation("./test_data/cases/simulation_1") ;
	) ;
}



TEST( Simulation, createFrom_simulation_1_GPU )
{
	Simulation * simulation = NULL ;

	ASSERT_NO_THROW
	(	
		simulation =  new Simulation("./test_data/cases/simulation_1_GPU") ;
	) ;

	Settings * settings = simulation->getSettings() ;

	EXPECT_EQ( "D3Q19", settings->getLatticeArrangementName() ) ;
	EXPECT_EQ( "incompressible", settings->getFluidModelName() ) ;
	EXPECT_EQ( "BGK", settings->getCollisionModelName() ) ;
	EXPECT_EQ( "double", settings->getDataTypeName() ) ;
	EXPECT_EQ( "GPU", settings->getComputationalEngineName() ) ;
	EXPECT_TRUE  (settings->isGeometryDefinedByPng()) ;
	EXPECT_FALSE (settings->isGeometryDefinedByVti()) ;

	ASSERT_NO_THROW( delete simulation ) ;
}



TEST( Simulation, createFrom_simulation_2_GPU )
{
	Simulation * simulation = NULL ;

	ASSERT_NO_THROW
	(	
		simulation =  new Simulation("./test_data/cases/simulation_2_GPU") ;
	) ;

	Settings * settings = simulation->getSettings() ;

	EXPECT_EQ( "D3Q19", settings->getLatticeArrangementName() ) ;
	EXPECT_EQ( "incompressible", settings->getFluidModelName() ) ;
	EXPECT_EQ( "BGK", settings->getCollisionModelName() ) ;
	EXPECT_EQ( "double", settings->getDataTypeName() ) ;
	EXPECT_EQ( "GPU", settings->getComputationalEngineName() ) ;
	EXPECT_TRUE  (settings->isGeometryDefinedByPng()) ;
	EXPECT_FALSE (settings->isGeometryDefinedByVti()) ;

	ASSERT_NO_THROW( delete simulation ) ;
}



TEST( Simulation, createFrom_simulation_3_GPU )
{
	Simulation * simulation = NULL ;

	ASSERT_NO_THROW
	(	
		simulation =  new Simulation("./test_data/cases/simulation_3_GPU") ;
	) ;

	Settings * settings = simulation->getSettings() ;

	EXPECT_EQ( "D3Q19", settings->getLatticeArrangementName() ) ;
	EXPECT_EQ( "incompressible", settings->getFluidModelName() ) ;
	EXPECT_EQ( "BGK", settings->getCollisionModelName() ) ;
	EXPECT_EQ( "double", settings->getDataTypeName() ) ;
	EXPECT_EQ( "GPU", settings->getComputationalEngineName() ) ;
	EXPECT_TRUE  (settings->isGeometryDefinedByPng()) ;
	EXPECT_FALSE (settings->isGeometryDefinedByVti()) ;

	ASSERT_NO_THROW( delete simulation ) ;
}



TEST( Simulation, createFrom_simulation_4_GPU )
{
	Simulation * simulation = NULL ;

	ASSERT_NO_THROW
	(	
		simulation =  new Simulation("./test_data/cases/simulation_4_GPU") ;
	) ;

	Settings * settings = simulation->getSettings() ;

	EXPECT_EQ( "D3Q19", settings->getLatticeArrangementName() ) ;
	EXPECT_EQ( "quasi_compressible", settings->getFluidModelName() ) ;
	EXPECT_EQ( "MRT", settings->getCollisionModelName() ) ;
	EXPECT_EQ( "double", settings->getDataTypeName() ) ;
	EXPECT_EQ( "GPU", settings->getComputationalEngineName() ) ;
	EXPECT_TRUE  (settings->isGeometryDefinedByPng()) ;
	EXPECT_FALSE (settings->isGeometryDefinedByVti()) ;

	ASSERT_NO_THROW( delete simulation ) ;
}



TEST (Simulation, createFrom_cross_200x200x200)
{
	Simulation * simulation = NULL ;

	ASSERT_NO_THROW
	(	
		simulation =  new Simulation("./test_data/cases/cross/200x200x200") ;
	) ;

	Settings * settings = simulation->getSettings() ;

	EXPECT_EQ( "D3Q19", settings->getLatticeArrangementName() ) ;
	EXPECT_EQ( "double", settings->getDataTypeName() ) ;
	EXPECT_EQ( "GPU", settings->getComputationalEngineName() ) ;
	EXPECT_FALSE (settings->isGeometryDefinedByPng()) ;
	EXPECT_TRUE  (settings->isGeometryDefinedByVti()) ;

	ASSERT_NO_THROW( delete simulation ) ;
}



void checkNode( const Simulation::NodeLB & node,
								NodeBaseType baseType,
								double rho, double rhoBoundary,
								const double (&u)[3], const double (&uBoundary)[3] )
{
	Coordinates c = node.coordinates ;

	ASSERT_EQ( 3u, node.u.size() ) << "difference at " << c << "\n" ;
	ASSERT_EQ( 3u, node.uBoundary.size() ) << "difference at " << c << "\n" ;
	ASSERT_EQ( 3u, node.uT0.size() ) << "difference at " << c << "\n" ;
	ASSERT_EQ( 19u, node.f.size() ) << "difference at " << c << "\n" ;
	ASSERT_EQ( 19u, node.fPost.size() ) << "difference at " << c << "\n" ;

	ASSERT_EQ( baseType, node.baseType ) << "difference at " << c << "\n" ;

	ASSERT_EQ( rho, node.rho ) << "difference at " << c << "\n" ;
	ASSERT_EQ( rhoBoundary, node.rhoBoundary ) << "difference at " << c << "\n" ;
	for (unsigned i=0 ; i < 3 ; i++)
	{
		ASSERT_EQ( u[i], node.u[i] ) << "difference at " << c << " for coordinate i=" << i << "\n" ;
		ASSERT_EQ( uBoundary[i], node.uBoundary[i] ) << "difference at " << c << " for coordinate i=" << i << "\n" ;
	}
}



void checkNode( const Simulation::NodeLB & node,
								NodeBaseType baseType,
								PlacementModifier placementModifier,
								double rho, double rhoBoundary,
								const double (&u)[3], const double (&uBoundary)[3] )
{
	ASSERT_EQ( placementModifier, node.placementModifier ) 
					<< "difference at " << node.coordinates << "\n" ;

	checkNode (node, baseType, rho, rhoBoundary, u, uBoundary) ;
}




void testGetNode_simulation_1(const std::string casePath )
{
	Simulation simulation( casePath ) ;

	auto settings = simulation.getSettings() ;

	// Nodes inside geometry
	EXPECT_NO_THROW( simulation.getNode( Coordinates(0,0,0) ) ) ;
	EXPECT_NO_THROW( simulation.getNode( Coordinates(3,3,3) ) ) ;
	// Nodes outside geometry
	EXPECT_NO_THROW( simulation.getNode( Coordinates(0,0,4) ) ) ;
	EXPECT_NO_THROW( simulation.getNode( Coordinates(0,4,0) ) ) ;
	EXPECT_NO_THROW( simulation.getNode( Coordinates(4,0,0) ) ) ;


	// Corners
	checkNode( simulation.getNode( Coordinates(0,0,0)), 
						NodeBaseType::BOUNCE_BACK_2, 0,1, {0,0,0}, {0,0,0} ) ;
	checkNode( simulation.getNode( Coordinates(0,3,0)), 
						NodeBaseType::BOUNCE_BACK_2, 0,1, {0,0,0}, {0,0,0} ) ;
	//checkNode( simulation.getNode( Coordinates(3,0,0)), 
	//					NodeBaseType::BOUNCE_BACK_2 ) ;
	//checkNode( simulation.getNode( Coordinates(3,3,0)), 
	//					NodeBaseType::BOUNCE_BACK_2 ) ;
	//checkNode( simulation.getNode( Coordinates(0,0,3)), 
	//					NodeBaseType::BOUNCE_BACK_2 ) ;
	//checkNode( simulation.getNode( Coordinates(0,3,3)), 
	//					NodeBaseType::BOUNCE_BACK_2 ) ;
	//checkNode( simulation.getNode( Coordinates(3,0,3)), 
	//					NodeBaseType::BOUNCE_BACK_2 ) ;
	checkNode( simulation.getNode( Coordinates(3,3,3)), 
						NodeBaseType::BOUNCE_BACK_2, 0,1, {0,0,0}, {0,0,0} ) ;

	double inletVelocity[3] = {0,0,0} ;
	inletVelocity[0] = settings->transformVelocityPhysicalToLB( 0.01 ) ;

	/*
		Remember, that covers (bottom and top for z=min and z=max) are treated specially.
	*/

	// Edges
	checkNode( simulation.getNode( Coordinates(0,0,1)), 
						NodeBaseType::BOUNCE_BACK_2, 0,1, {0,0,0}, inletVelocity ) ;
	checkNode( simulation.getNode( Coordinates(0,0,2)), 
						NodeBaseType::BOUNCE_BACK_2, 0,1, {0,0,0}, inletVelocity ) ;
	checkNode( simulation.getNode( Coordinates(0,1,0)), 
						NodeBaseType::BOUNCE_BACK_2, 0,1, {0,0,0}, {0,0,0} ) ;
	checkNode( simulation.getNode( Coordinates(0,2,0)), 
						NodeBaseType::BOUNCE_BACK_2, 0,1, {0,0,0}, {0,0,0} ) ;

	// Inlet
	checkNode( simulation.getNode( Coordinates(0,1,1)), 
						NodeBaseType::VELOCITY, 0,1, {0,0,0}, inletVelocity ) ;


	// Channel
	checkNode( simulation.getNode( Coordinates(1,2,1)), 
						NodeBaseType::FLUID, 0,1, {0,0,0}, {0,0,0} ) ;
	checkNode( simulation.getNode( Coordinates(1,1,1)), 
						NodeBaseType::BOUNCE_BACK_2, 0,1, {0,0,0}, {0,0,0} ) ;
}



TEST( Simulation, getNode )
{
	testGetNode_simulation_1("./test_data/cases/simulation_1") ;
}



TEST( Simulation, getNode_GPU )
{
	testGetNode_simulation_1("./test_data/cases/simulation_1_GPU") ;
}



void testSimulationGeometryModifiers (const std::string engine)
{
	Simulation * simulation = NULL ;

	ASSERT_NO_THROW
	(	
		simulation =  new Simulation("./test_data/cases/simulation_geometry_modifiers_" + engine) ;
	) ;

	Settings * settings = simulation->getSettings() ;

	EXPECT_EQ( "D3Q19", settings->getLatticeArrangementName() ) ;
	EXPECT_EQ( "quasi_compressible", settings->getFluidModelName() ) ;
	EXPECT_EQ( "MRT", settings->getCollisionModelName() ) ;
	EXPECT_EQ( "double", settings->getDataTypeName() ) ;
	EXPECT_EQ( engine, settings->getComputationalEngineName() ) ;

	// initialGeometryModificator.rb
	checkNode( simulation->getNode( Coordinates(30,30,30)), 
						NodeBaseType::FLUID, PlacementModifier::NONE, 
						settings->transformPressurePhysicalToVolumetricMassDensityLB (0.9), 
						settings->transformPressurePhysicalToVolumetricMassDensityLB (0.8), 
						{
							settings->transformVelocityPhysicalToLB (2),
							settings->transformVelocityPhysicalToLB (2),
							settings->transformVelocityPhysicalToLB (2)
						}, 
						{
							settings->transformVelocityPhysicalToLB (3),
							settings->transformVelocityPhysicalToLB (3),
							settings->transformVelocityPhysicalToLB (3)
						} ) ;

	//finalGeometryModificator.rb
	checkNode( simulation->getNode( Coordinates(31,31,31)), 
						NodeBaseType::VELOCITY, PlacementModifier::TOP, 
						settings->transformPressurePhysicalToVolumetricMassDensityLB (0.5), 
						settings->transformPressurePhysicalToVolumetricMassDensityLB (0.4), 
						{
							settings->transformVelocityPhysicalToLB (4),
							settings->transformVelocityPhysicalToLB (4),
							settings->transformVelocityPhysicalToLB (4)
						}, 
						{
							settings->transformVelocityPhysicalToLB (5),
							settings->transformVelocityPhysicalToLB (5),
							settings->transformVelocityPhysicalToLB (5)
						} ) ;

	ASSERT_NO_THROW (simulation->run()) ; // Saves vtk files with geometry.

	ASSERT_NO_THROW( delete simulation ) ;
}



TEST( Simulation, simulation_geometry_modifiers_CPU )
{
	testSimulationGeometryModifiers ("CPU") ;
}



TEST( Simulation, simulation_geometry_modifiers_GPU )
{
	testSimulationGeometryModifiers ("GPU") ;
}

