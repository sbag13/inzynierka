#include "gtest/gtest.h"

#include "TilingStatistic.hpp"
#include "TileLayout.hpp"
#include "TileDefinitions.hpp"

#include "NodeLayoutTest.hpp"



using namespace microflow ;
using namespace std ;



TEST( TilingStatistic, constructor0 )
{
	ASSERT_NO_THROW( TilingStatistic tilingStatistic ; ) ;
}


TEST( TilingStatistic, constructor )
{
	NodeLayout nodeLayout = createSolidNodeLayout( 32, 32, 16) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	ASSERT_NO_THROW( tileLayout.computeTilingStatistic() ) ;
}


TEST( TilingStatistic, computeNonEmptyTilesFactor_1 )
{
	TilingStatistic tilingStatistic ;

	tilingStatistic.setNTotalTiles(2) ;
	EXPECT_EQ( tilingStatistic.computeNonEmptyTilesFactor(), 0.0 ) ;

	tilingStatistic.increaseNonemptyTilesCounter() ;
	EXPECT_EQ( tilingStatistic.computeNonEmptyTilesFactor(), 0.5 ) ;

	tilingStatistic.increaseNonemptyTilesCounter() ;
	EXPECT_EQ( tilingStatistic.computeNonEmptyTilesFactor(), 1.0 ) ;
}



TEST( TilingStatistic, getNFluidNodes )
{
	TilingStatistic tilingStatistic ;

	EXPECT_EQ( tilingStatistic.getNFluidNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), size_t(1) ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), size_t(2) ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), size_t(3) ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), size_t(3) ) ;

	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), size_t(3) ) ;
}



TEST( TilingStatistic, getNBoundaryNodes )
{
	TilingStatistic tilingStatistic ;

	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), size_t(1) ) ;

	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), size_t(2) ) ;

	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), size_t(3) ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), size_t(3) ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), size_t(3) ) ;
}



TEST( TilingStatistic, getNUnknownNodes )
{
	TilingStatistic tilingStatistic ;

	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::MARKER ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(1) ) ;

	tilingStatistic.addNode( NodeBaseType::MARKER ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(2) ) ;

	tilingStatistic.addNode( NodeBaseType::MARKER ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(3) ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(3) ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(3) ) ;

	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), size_t(3) ) ;
}



TEST( TilingStatistic, getNSolidNodesInTiles )
{
	TilingStatistic tilingStatistic ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), size_t(0) ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	tilingStatistic.addNode( NodeBaseType::MARKER ) ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 0u ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 1u ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 2u ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 3u ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	tilingStatistic.addNode( NodeBaseType::MARKER ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 3u ) ;
}



TEST( TilingStatistic, getNSolidNodesInTotal_emptyTile )
{
	TilingStatistic tilingStatistic ;

	size_t nodesPerTile = DEFAULT_3D_TILE_EDGE * DEFAULT_3D_TILE_EDGE * DEFAULT_3D_TILE_EDGE ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 0u ) ;

	tilingStatistic.setNTotalTiles( 1 ) ;	
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), nodesPerTile ) ;

	tilingStatistic.setNTotalTiles( 2 ) ;	
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 2 * nodesPerTile ) ;

	tilingStatistic.setNTotalTiles( 3 ) ;	
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 3 * nodesPerTile ) ;
}



TEST( TilingStatistic, getNSolidNodesInTotal )
{
	TilingStatistic tilingStatistic ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 0u ) ;

	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	tilingStatistic.addNode( NodeBaseType::SOLID ) ;
	tilingStatistic.addNode( NodeBaseType::SOLID ) ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 3u ) ;

	tilingStatistic.increaseNonemptyTilesCounter() ;
	tilingStatistic.setNTotalTiles( 1 ) ;

	size_t nodesPerTile = DEFAULT_3D_TILE_EDGE * DEFAULT_3D_TILE_EDGE * DEFAULT_3D_TILE_EDGE ;
	
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 3u ) ;

	tilingStatistic.setNTotalTiles( 2 ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 3 + nodesPerTile ) ;

	tilingStatistic.increaseNonemptyTilesCounter() ;
	tilingStatistic.setNTotalTiles( 2 ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 3u ) ;

	tilingStatistic.setNTotalTiles( 4 ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 3 + 2*nodesPerTile ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;
	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	tilingStatistic.addNode( NodeBaseType::MARKER ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 3 + 2*nodesPerTile ) ;
}



TEST( TilingStatistic, addNode )
{
	TilingStatistic tilingStatistic ;
	
	tilingStatistic.addNode( NodeBaseType::SOLID ) ;

	tilingStatistic.addNode( NodeBaseType::FLUID ) ;

	tilingStatistic.addNode( NodeBaseType::BOUNCE_BACK_2 ) ;
	tilingStatistic.addNode( NodeBaseType::VELOCITY ) ;
	tilingStatistic.addNode( NodeBaseType::VELOCITY_0 ) ;
	tilingStatistic.addNode( NodeBaseType::PRESSURE ) ;

	tilingStatistic.addNode( NodeBaseType::MARKER ) ;
	tilingStatistic.addNode( NodeBaseType::SIZE ) ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), 4u ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), 2u ) ;
}



TEST( TilingStatistic, computeNonEmptyTilesFactor_noTiles )
{
	NodeLayout nodeLayout = createSolidNodeLayout( 4, 4, 4) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

	EXPECT_EQ( tilingStatistic.getNNonEmptyTiles(), 0u ) ;
	EXPECT_EQ( tilingStatistic.getNEmptyTiles(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNTotalTiles(), 1u ) ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 0u ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 64u ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), 0u ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), 0u ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), 0u ) ;
}



TEST( TilingStatistic, computeNonEmptyTilesFactor_singleTileFluid )
{
	NodeLayout nodeLayout = createSolidNodeLayout( 4, 4, 4) ;

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::FLUID) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

	EXPECT_EQ( tilingStatistic.getNNonEmptyTiles(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNEmptyTiles(), 0u ) ;
	EXPECT_EQ( tilingStatistic.getNTotalTiles(), 1u ) ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 63u ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 63u ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), 0u ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), 0u ) ;
}



TEST( TilingStatistic, computeNonEmptyTilesFactor_singleTileFluid_manySolidTiles )
{
	NodeLayout nodeLayout = createSolidNodeLayout( 32, 32, 16) ;

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::FLUID) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

	EXPECT_EQ( tilingStatistic.getNNonEmptyTiles(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNEmptyTiles(), 255u ) ;
	EXPECT_EQ( tilingStatistic.getNTotalTiles(), 256u ) ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 63u ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 256u*64u - 1u ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), 0u ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), 0u ) ;
}



TEST( TilingStatistic, computeNonEmptyTilesFactor_singleTileBoundary_manySolidTiles )
{
	NodeLayout nodeLayout = createSolidNodeLayout( 32, 32, 16) ;

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::BOUNCE_BACK_2) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

	EXPECT_EQ( tilingStatistic.getNNonEmptyTiles(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNEmptyTiles(), 255u ) ;
	EXPECT_EQ( tilingStatistic.getNTotalTiles(), 256u ) ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 63u ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 256u*64u - 1u ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), 0u ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), 0u ) ;
}



TEST( TilingStatistic, computeNonEmptyTilesFactor_singleTileManyNodes_manySolidTiles )
{
	NodeLayout nodeLayout = createSolidNodeLayout( 32, 32, 16) ;

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType(1, 1, 2, NodeBaseType::BOUNCE_BACK_2) ;
	nodeLayout.setNodeType(1, 2, 1, NodeBaseType::VELOCITY) ;
	nodeLayout.setNodeType(2, 1, 1, NodeBaseType::VELOCITY_0) ;
	nodeLayout.setNodeType(2, 2, 1, NodeBaseType::PRESSURE) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

	EXPECT_EQ( tilingStatistic.getNNonEmptyTiles(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNEmptyTiles(), 255u ) ;
	EXPECT_EQ( tilingStatistic.getNTotalTiles(), 256u ) ;

	EXPECT_EQ( tilingStatistic.getNSolidNodesInTiles(), 64u-5u ) ;
	EXPECT_EQ( tilingStatistic.getNSolidNodesInTotal(), 256u*64u - 5u ) ;
	EXPECT_EQ( tilingStatistic.getNFluidNodes(), 1u ) ;
	EXPECT_EQ( tilingStatistic.getNBoundaryNodes(), 4u ) ;
	EXPECT_EQ( tilingStatistic.getNUnknownNodes(), 0u ) ;
}



TEST( TilingStatistic, computeAverageTileUtilisation_NoTiles )
{
	NodeLayout nodeLayout = createSolidNodeLayout( 32, 32, 16) ;

	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

		ASSERT_EQ( tilingStatistic.computeAverageTileUtilisation(), 0 ) ;
	}

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::FLUID) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;
		
		double expectedUtilization = 1.0 / TilingStatistic::getNodesPerTile() ;
		ASSERT_EQ( tilingStatistic.computeAverageTileUtilisation(), expectedUtilization ) ;
	}

	nodeLayout.setNodeType(1, 1, 2, NodeBaseType::PRESSURE) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;
		
		double expectedUtilization = 2.0 / TilingStatistic::getNodesPerTile() ;
		ASSERT_EQ( tilingStatistic.computeAverageTileUtilisation(), expectedUtilization ) ;
	}

	nodeLayout = createFluidNodeLayout( 32, 32, 16 ) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;
		
		double expectedUtilization = 1.0 ;
		ASSERT_EQ( tilingStatistic.computeAverageTileUtilisation(), expectedUtilization ) ;
	}
}



TEST( TilingStatistic, computeGeometryDensity_singleTile )
{
	const size_t width = DEFAULT_3D_TILE_EDGE ;
	const size_t height = DEFAULT_3D_TILE_EDGE ;
	const size_t depth = DEFAULT_3D_TILE_EDGE ;

	NodeLayout nodeLayout = createSolidNodeLayout( width, height, depth) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

		EXPECT_EQ( tilingStatistic.computeGeometryDensity(), 0 ) ;

		std::cout << tilingStatistic.computeStatistics() << "\n" ;
	}

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::FLUID) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

		EXPECT_EQ( tilingStatistic.computeGeometryDensity(), 
								1.0 / TileLayout<StorageOnCPU>::NonEmptyTile::getNNodesPerTile() ) ;
	}

	nodeLayout.setNodeType(1, 1, 2, NodeBaseType::PRESSURE) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

		EXPECT_EQ( tilingStatistic.computeGeometryDensity(), 
								2.0 / TileLayout<StorageOnCPU>::NonEmptyTile::getNNodesPerTile() ) ;
	}

	nodeLayout = createFluidNodeLayout( width, height, depth) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

		EXPECT_EQ( tilingStatistic.computeGeometryDensity(), 1.0 ) ;
	}
}



TEST( TilingStatistic, computeGeometryDensity_manyTiles )
{
	const size_t width = 8 ;
	const size_t height = 8 ;
	const size_t depth = 8 ;

	
	const size_t nTiles = ( width/DEFAULT_3D_TILE_EDGE) * 
												( height/DEFAULT_3D_TILE_EDGE) *
												( depth/DEFAULT_3D_TILE_EDGE) ;

	NodeLayout nodeLayout = createSolidNodeLayout( width, height, depth) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

		ASSERT_EQ( tilingStatistic.computeGeometryDensity(), 0 ) ;
	}

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::FLUID) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

		ASSERT_EQ( tilingStatistic.computeGeometryDensity(), 
								1.0 / (TileLayout<StorageOnCPU>::NonEmptyTile::getNNodesPerTile() * nTiles ) );
	}

	nodeLayout = createFluidNodeLayout(width, height, depth) ;
	{
		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
		TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;

		ASSERT_EQ( tilingStatistic.computeGeometryDensity(), 1.0 ) ;
	}
}



