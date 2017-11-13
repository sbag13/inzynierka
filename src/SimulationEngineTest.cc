#include "gtest/gtest.h"



#include <iostream>



#include "SimulationEngine.hpp"
#include "TileLayoutTest.hpp"



using namespace microflow ;
using namespace std ;



TEST( SimulationEngineFactory, create_D3Q19_incompressible_BGK_double_CPU )
{
	ASSERT_NO_THROW
	(
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

		// FIXME: maybe ExpandedNodeLayout should be generated inside SimulationEngineFactory ?
		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		SimulationEngine * simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"D3Q19_incompressible_BGK_double_CPU", Settings(), tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelIncompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelBGK", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "CPU", simulationEngine->getComputationalEngineName() ) ;

		delete simulationEngine ;
	) ;
}



TEST( SimulationEngineFactory, create_D3Q19_incompressible_BGK_double_GPU )
{
	ASSERT_NO_THROW
	(
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

		SimulationEngine * simulationEngine = NULL ;

		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"D3Q19_incompressible_BGK_double_GPU", Settings(), tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelIncompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelBGK", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "GPU", simulationEngine->getComputationalEngineName() ) ;

		delete simulationEngine ;
	) ;
}



TEST( SimulationEngineFactory, create_D3Q19_quasicompressible_BGK_double_CPU )
{
	ASSERT_NO_THROW
	(
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		SimulationEngine * simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"D3Q19_quasi_compressible_BGK_double_CPU", Settings(), tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelQuasicompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelBGK", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "CPU", simulationEngine->getComputationalEngineName() ) ;

		delete simulationEngine ;
	) ;
}



TEST( SimulationEngineFactory, create_D3Q19_quasicompressible_BGK_double_GPU )
{
	ASSERT_NO_THROW
	(
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		SimulationEngine * simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"D3Q19_quasi_compressible_BGK_double_GPU", Settings(), tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelQuasicompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelBGK", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "GPU", simulationEngine->getComputationalEngineName() ) ;

		delete simulationEngine ;
	) ;
}



TEST( SimulationEngineFactory, create_D3Q19_incompressible_MRT_double_CPU )
{
	ASSERT_NO_THROW
	(
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		SimulationEngine * simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"D3Q19_incompressible_MRT_double_CPU", Settings(), tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelIncompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelMRT", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "CPU", simulationEngine->getComputationalEngineName() ) ;

		delete simulationEngine ;
	) ;
}



TEST( SimulationEngineFactory, create_D3Q19_incompressible_MRT_double_GPU )
{
	ASSERT_NO_THROW
	(
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

		SimulationEngine * simulationEngine = NULL ;

		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"D3Q19_incompressible_MRT_double_GPU", Settings(), tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelIncompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelMRT", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "GPU", simulationEngine->getComputationalEngineName() ) ;

		delete simulationEngine ;
	) ;
}



TEST( SimulationEngineFactory, create_D3Q19_quasicompressible_MRT_double_CPU )
{
	ASSERT_NO_THROW
	(
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		SimulationEngine * simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"D3Q19_quasi_compressible_MRT_double_CPU", Settings(), tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelQuasicompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelMRT", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "CPU", simulationEngine->getComputationalEngineName() ) ;

		delete simulationEngine ;
	) ;
}



TEST( SimulationEngineFactory, create_D3Q19_quasicompressible_MRT_double_GPU )
{
	ASSERT_NO_THROW
	(
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		SimulationEngine * simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"D3Q19_quasi_compressible_MRT_double_GPU", Settings(), tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelQuasicompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelMRT", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "GPU", simulationEngine->getComputationalEngineName() ) ;

		delete simulationEngine ;
	) ;
}



TEST( SimulationEngineFactory, createFromSettings_D3Q19_incompressible_BGK_double_CPU )
{
	ASSERT_NO_THROW
	(
		Settings settings("./test_data/cases/configuration_3") ;
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;
	
		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		SimulationEngine * simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				settings, tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelIncompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelBGK", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "CPU", simulationEngine->getComputationalEngineName() ) ;
	) ;
}



TEST( SimulationEngineFactory, createFromSettings_D3Q19_quasicompressible_MRT_double_GPU )
{
	ASSERT_NO_THROW
	(
		Settings settings("./test_data/cases/configuration_4") ;
		TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;
		
		NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;

		SimulationEngine * simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				settings, tileLayout, expandedNodeLayout
			) ;

		EXPECT_EQ( "D3Q19", simulationEngine->getLatticeArrangementName() ) ;
		EXPECT_EQ( "FluidModelQuasicompressible", simulationEngine->getFluidModelName() ) ;
		EXPECT_EQ( "CollisionModelMRT", simulationEngine->getCollisionModelName() ) ;
		EXPECT_EQ( "double", simulationEngine->getDataTypeName() ) ;
		EXPECT_EQ( "GPU", simulationEngine->getComputationalEngineName() ) ;
	) ;
}



TEST( SimulationEngineFactory, create_nonExistent )
{
	TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( 4, 4, 4 ) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;


	ASSERT_ANY_THROW 
	(
		auto simulationEngine = 
			SimulationEngineFactory::createEngine
			(
				"NON_EXISTENT_DEFINITION", Settings(), tileLayout, expandedNodeLayout
			) ;
			delete simulationEngine ;
	) ;
}
