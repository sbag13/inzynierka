#include "gtest/gtest.h"



#include "Settings.hpp"



using namespace microflow ;



TEST(Settings, no_configuration)
{
	EXPECT_ANY_THROW( Settings settings("")) ;
}



TEST(Settings, configuration_3)
{
	Settings * settings = NULL ;

	EXPECT_NO_THROW( settings = new Settings("./test_data/cases/configuration_3")) ;
	EXPECT_EQ(settings->getFluidModelName(),"incompressible") ;
	EXPECT_EQ(settings->getLatticeArrangementName(),"D3Q19") ;
	EXPECT_EQ(settings->getDataTypeName(),"double") ;
	EXPECT_EQ(settings->getCollisionModelName(),"BGK") ;
	EXPECT_EQ(settings->getComputationalEngineName(),"CPU") ;
	EXPECT_EQ(settings->getZExpandDepth(), 1u) ;

  EXPECT_EQ(settings->getDefaultWallNode()                   , 
		NodeType(NodeBaseType::VELOCITY_0)) ;
  EXPECT_EQ(settings->getDefaultExternalCornerNode()         , 
		NodeType(NodeBaseType::VELOCITY_0, PlacementModifier::EXTERNAL_CORNER)) ;
  EXPECT_EQ(settings->getDefaultInternalCornerNode()         , 
		NodeType(NodeBaseType::FLUID)) ;
  EXPECT_EQ(settings->getDefaultExternalEdgeNode()           , 
		NodeType(NodeBaseType::VELOCITY_0, PlacementModifier::EXTERNAL_EDGE)) ;
  EXPECT_EQ(settings->getDefaultInternalEdgeNode()           , 
		NodeType(NodeBaseType::VELOCITY_0, PlacementModifier::INTERNAL_EDGE)) ;
  EXPECT_EQ(settings->getDefaultNotIdentifiedNode()          , 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;
  EXPECT_EQ(settings->getDefaultExternalEdgePressureNode()   , 
		NodeType(NodeBaseType::VELOCITY_0, PlacementModifier::EXTERNAL_EDGE_PRESSURE_TANGENTIAL)) ;
  EXPECT_EQ(settings->getDefaultExternalCornerPressureNode() , 
		NodeType(NodeBaseType::VELOCITY_0, PlacementModifier::EXTERNAL_CORNER_PRESSURE_TANGENTIAL)) ;
  EXPECT_EQ(settings->getDefaultEdgeToPerpendicularWallNode(), 
		NodeType(NodeBaseType::VELOCITY_0, PlacementModifier::CORNER_ON_EDGE_AND_PERPENDICULAR_PLANE)) ;

	delete settings ; settings = NULL ;
}



TEST(Settings, simulation_1)
{
	Settings * settings = NULL ;

	EXPECT_NO_THROW( settings = new Settings("./test_data/cases/simulation_1")) ;
	EXPECT_EQ(settings->getFluidModelName(),"incompressible") ;
	EXPECT_EQ(settings->getLatticeArrangementName(),"D3Q19") ;
	EXPECT_EQ(settings->getDataTypeName(),"double") ;
	EXPECT_EQ(settings->getCollisionModelName(),"BGK") ;
	EXPECT_EQ(settings->getComputationalEngineName(),"CPU") ;
	EXPECT_EQ(settings->getZExpandDepth(), 4u) ;

  EXPECT_EQ(settings->getDefaultWallNode()                   , 
		NodeType(NodeBaseType::VELOCITY_0)) ;
  EXPECT_EQ(settings->getDefaultExternalCornerNode()         , 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;
  EXPECT_EQ(settings->getDefaultInternalCornerNode()         , 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;
  EXPECT_EQ(settings->getDefaultExternalEdgeNode()           , 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;
  EXPECT_EQ(settings->getDefaultInternalEdgeNode()           , 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;
  EXPECT_EQ(settings->getDefaultNotIdentifiedNode()          , 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;
  EXPECT_EQ(settings->getDefaultExternalEdgePressureNode()   , 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;
  EXPECT_EQ(settings->getDefaultExternalCornerPressureNode() , 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;
  EXPECT_EQ(settings->getDefaultEdgeToPerpendicularWallNode(), 
		NodeType(NodeBaseType::BOUNCE_BACK_2)) ;

	delete settings ; settings = NULL ;
}



TEST (Settings, cross_200x200x200)
{
	Settings * settings = NULL ;

	EXPECT_NO_THROW (settings = new Settings("./test_data/cases/cross/200x200x200")) ;

	EXPECT_EQ ("D3Q19", settings->getLatticeArrangementName()) ;
	EXPECT_EQ ("double", settings->getDataTypeName()) ;
	EXPECT_EQ ("GPU", settings->getComputationalEngineName()) ;
	EXPECT_FALSE (settings->isGeometryDefinedByPng()) ;
	EXPECT_TRUE  (settings->isGeometryDefinedByVti()) ;

	ASSERT_NO_THROW (delete settings) ;
}



TEST (Settings, geometryModificators)
{
	Settings * settings = NULL ;

	EXPECT_NO_THROW( settings = new Settings("./test_data/cases/ruby_geometry_modifiers")) ;

	NodeLayout nodeLayout 
		(
			ColoredPixelClassificator (settings->getPixelColorDefinitionsFilePath()),
			Image (settings->getGeometryPngImagePath()),
			settings->getZExpandDepth()
		) ;

	EXPECT_EQ (nodeLayout.getNodeType(1,1,1), NodeBaseType::FLUID) ;

	settings->initialModify (nodeLayout) ;

	EXPECT_EQ (nodeLayout.getNodeType(1,1,1), NodeBaseType::SOLID) ;
	EXPECT_EQ (nodeLayout.getNodeType(2,2,2), NodeBaseType::FLUID) ;

	
	settings->finalModify (nodeLayout) ;

	EXPECT_EQ (nodeLayout.getNodeType(2,2,2), NodeBaseType::SOLID) ;

	delete settings ;
}



TEST (Settings, geometryModificators_shapes)
{
	Settings * settings = NULL ;

	EXPECT_NO_THROW( settings = new Settings("./test_data/cases/ruby_geometry_modifiers_shapes")) ;
	
	ASSERT_NO_THROW
	(
		NodeLayout nodeLayout 
			(
				ColoredPixelClassificator (settings->getPixelColorDefinitionsFilePath()),
				Image (settings->getGeometryPngImagePath()),
				settings->getZExpandDepth()
			) ;

		EXPECT_EQ (Size(40,40,40), nodeLayout.getSize()) ;

		EXPECT_EQ (nodeLayout.getNodeType(1,1,1), NodeBaseType::SOLID) ;

		settings->initialModify (nodeLayout) ;

		EXPECT_EQ (nodeLayout.getNodeType(1,1,1), NodeBaseType::FLUID) ;
		EXPECT_EQ (nodeLayout.getNodeType(2,2,2), NodeBaseType::SOLID) ;

		
		settings->finalModify (nodeLayout) ;

		EXPECT_EQ (nodeLayout.getNodeType(1,1,1), NodeBaseType::SOLID) ;

		// myShape method
		EXPECT_EQ (nodeLayout.getNodeType(0,0,0), NodeBaseType::PRESSURE) ;
		
		/*
			Filled box at (3,3,3), (4,4,4)
		*/
		// Nodes outside close to walls
		EXPECT_EQ (nodeLayout.getNodeType(2,2,2), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(3,3,2), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(5,5,5), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(5,4,4), NodeBaseType::SOLID) ;
		// All box nodes
		EXPECT_EQ (nodeLayout.getNodeType(3,3,3), NodeBaseType::VELOCITY) ;
		EXPECT_EQ (nodeLayout.getNodeType(3,3,4), NodeBaseType::VELOCITY) ;
		EXPECT_EQ (nodeLayout.getNodeType(3,4,3), NodeBaseType::VELOCITY) ;
		EXPECT_EQ (nodeLayout.getNodeType(3,4,4), NodeBaseType::VELOCITY) ;
		EXPECT_EQ (nodeLayout.getNodeType(4,3,3), NodeBaseType::VELOCITY) ;
		EXPECT_EQ (nodeLayout.getNodeType(4,3,4), NodeBaseType::VELOCITY) ;
		EXPECT_EQ (nodeLayout.getNodeType(4,4,3), NodeBaseType::VELOCITY) ;
		EXPECT_EQ (nodeLayout.getNodeType(4,4,4), NodeBaseType::VELOCITY) ;

		/*
			Empty box at (6,6,6), (9,9,9)
		*/
		// Interior of the box
		EXPECT_EQ (nodeLayout.getNodeType(7,7,7), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,7,8), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,8,7), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,8,8), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,7,7), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,7,8), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,8,7), NodeBaseType::SOLID) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,8,8), NodeBaseType::SOLID) ;
		// Corners
		EXPECT_EQ (nodeLayout.getNodeType(6,6,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,9,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,6,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,9,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,6,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,9,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,6,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,9,9), NodeBaseType::VELOCITY_0) ;
		// Edges along Z
		EXPECT_EQ (nodeLayout.getNodeType(6,6,7), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,6,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,9,7), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,9,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,6,7), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,6,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,9,7), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,9,8), NodeBaseType::VELOCITY_0) ;
		// Edges along Y
		EXPECT_EQ (nodeLayout.getNodeType(6,7,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,8,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,7,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,8,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,7,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,8,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,7,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,8,9), NodeBaseType::VELOCITY_0) ;
		// Edges along X
		EXPECT_EQ (nodeLayout.getNodeType(7,6,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,6,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,6,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,6,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,9,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,9,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,9,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,9,9), NodeBaseType::VELOCITY_0) ;
		// Walls
		EXPECT_EQ (nodeLayout.getNodeType(7,6,7), NodeBaseType::VELOCITY_0) ; //bottom
		EXPECT_EQ (nodeLayout.getNodeType(7,6,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,6,7), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,6,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,9,7), NodeBaseType::VELOCITY_0) ; // top
		EXPECT_EQ (nodeLayout.getNodeType(7,9,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,9,7), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,9,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,7,7), NodeBaseType::VELOCITY_0) ; // left
		EXPECT_EQ (nodeLayout.getNodeType(6,7,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,8,7), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(6,8,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,7,7), NodeBaseType::VELOCITY_0) ; // right
		EXPECT_EQ (nodeLayout.getNodeType(9,7,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,8,7), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(9,8,8), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,7,6), NodeBaseType::VELOCITY_0) ; // front
		EXPECT_EQ (nodeLayout.getNodeType(7,8,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,7,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,8,6), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(7,7,9), NodeBaseType::VELOCITY_0) ; // back
		EXPECT_EQ (nodeLayout.getNodeType(7,8,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,7,9), NodeBaseType::VELOCITY_0) ;
		EXPECT_EQ (nodeLayout.getNodeType(8,8,9), NodeBaseType::VELOCITY_0) ;


		/*
			Ball (filled sphere) at (20,20,20) radius 8
		*/
		EXPECT_EQ (nodeLayout.getNodeType(20,20,20), NodeBaseType::PRESSURE) ;
		EXPECT_EQ (nodeLayout.getNodeType(20,20,28), NodeBaseType::PRESSURE) ;
		EXPECT_EQ (nodeLayout.getNodeType(20,28,20), NodeBaseType::PRESSURE) ;
		EXPECT_EQ (nodeLayout.getNodeType(28,20,20), NodeBaseType::PRESSURE) ;
		EXPECT_EQ (nodeLayout.getNodeType(20,20,12), NodeBaseType::PRESSURE) ;
		EXPECT_EQ (nodeLayout.getNodeType(20,12,20), NodeBaseType::PRESSURE) ;
		EXPECT_EQ (nodeLayout.getNodeType(12,20,20), NodeBaseType::PRESSURE) ;

		
		delete settings ;
	) ;
}

