#include "gtest/gtest.h"

#include <string>

#include "TileLayoutTest.hpp"
#include "Settings.hpp"
#include "PerformanceMeter.hpp"
#include "ExpandedNodeLayout.hpp"
#include "TestPath.hpp"



using namespace std ;



void testCopyTileLayoutGPU( TileLayout<StorageOnCPU> & tileLayout )
{
	ASSERT_NO_THROW( TileLayout< StorageOnGPU > tileLayoutGPU( tileLayout ) ) ;
	TileLayout< StorageOnGPU > tileLayoutGPU( tileLayout ) ;

	NodeLayout nodeLayout2 = createSolidNodeLayout(4, 4, 4) ;
	TileLayout<StorageOnCPU> tileLayout2( nodeLayout2 ) ;
	ASSERT_NO_THROW( tileLayoutGPU.copyToCPU( tileLayout2 ) ) ;
	
	EXPECT_TRUE( tileLayout == tileLayout2 ) ;
}



TEST( TileLayout, constructor )
{
	const size_t Nx = 21 ;
	const size_t Ny = 121 ;
	const size_t Nz = 11 ;

	TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout( Nx, Ny, Nz ) ;

	ASSERT_EQ( (24u/4u) * (124u/4u) * (12u/4u), tileLayout.computeNoTilesTotal() ) ;

	EXPECT_TRUE( tileLayout.isTileEmpty(0,0,0) ) ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, singleTile )
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	ASSERT_EQ( (24u/4u) * (124u/4u) * (12u/4u), tileLayout.computeNoTilesTotal() ) ;

	EXPECT_FALSE( tileLayout.isTileEmpty(0,0,0) ) ;

	EXPECT_TRUE( tileLayout.isTileEmpty(4,0,0) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(0,4,0) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(0,0,4) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(4,4,4) ) ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, twoTiles )
{
	const size_t Nx = 21 ;
	const size_t Ny = 121 ;
	const size_t Nz = 11 ;

	TileLayout<StorageOnCPU> tileLayout = generateTwoTilesLayout( Nx, Ny, Nz ) ;

	ASSERT_EQ( (24u/4u) * (124u/4u) * (12u/4u), tileLayout.computeNoTilesTotal() ) ;

	EXPECT_FALSE( tileLayout.isTileEmpty(0,0,0) ) ;
	EXPECT_FALSE( tileLayout.isTileEmpty(4,0,0) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(0,4,0) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(0,0,4) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(4,4,4) ) ;

	tileLayout.saveToVolFile(testOutDirectory + "tiles_twoTiles") ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, corners )
{
	const size_t Nx = 21 ;
	const size_t Ny = 121 ;
	const size_t Nz = 11 ;

	NodeLayout nodeLayout = createHomogenousNodeLayout( Nx, Ny, Nz, Pixel(255,255,255) ) ;

	nodeLayout.setNodeType( 1,   1,  1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType(20,   1,  1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType( 1, 120,  1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType(20, 120,  1, NodeBaseType::FLUID) ;

	nodeLayout.setNodeType( 1,   1,  10, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType(20,   1,  10, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType( 1, 120,  10, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType(20, 120,  10, NodeBaseType::FLUID) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	ASSERT_EQ( (24u/4u) * (124u/4u) * (12u/4u), tileLayout.computeNoTilesTotal() ) ;

	EXPECT_FALSE( tileLayout.isTileEmpty(0,0,0) ) ;
	EXPECT_FALSE( tileLayout.isTileEmpty(20,0,0) ) ;
	EXPECT_FALSE( tileLayout.isTileEmpty(0,120,0) ) ;
	EXPECT_FALSE( tileLayout.isTileEmpty(20,120,0) ) ;

	EXPECT_FALSE( tileLayout.isTileEmpty(0,0,8) ) ;
	EXPECT_FALSE( tileLayout.isTileEmpty(20,0,8) ) ;
	EXPECT_FALSE( tileLayout.isTileEmpty(0,120,8) ) ;
	EXPECT_FALSE( tileLayout.isTileEmpty(20,120,8) ) ;

	EXPECT_TRUE( tileLayout.isTileEmpty(0,0,4) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(20,0,4) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(0,120,4) ) ;
	EXPECT_TRUE( tileLayout.isTileEmpty(20,120,4) ) ;

	for (size_t z=0 ; z < 3 ; z ++ )
		for (size_t y=1 ; y < (124/4 - 1) ; y++ )
			for (size_t x=1 ; x < (24/4 - 1) ; x ++ )
			{
				ASSERT_TRUE( tileLayout.isTileEmpty(x*4, y*4, z*4) ) ;
			}

	tileLayout.saveToVolFile(testOutDirectory + "tiles_corners") ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, generateTiledLayout_no_tiles)
{
	size_t width = 16 ;
	size_t height = 16 ;
	size_t depth = 16 ;

	NodeLayout nodeLayout = createSolidNodeLayout(width, height, depth) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	NodeLayout tiledLayout = tileLayout.generateLayoutWithMarkedTiles() ;

	EXPECT_EQ (tiledLayout.getSize(), nodeLayout.getSize()) ;

	for (size_t z=0 ; z < depth ; z++)
		for (size_t y=0 ; y < height ; y++)
			for (size_t x=0 ; x < width ; x++)
			{
				NodeType node = tiledLayout.getNodeType(x, y, z) ;
				ASSERT_TRUE( node.isSolid() ) ;
				ASSERT_EQ  ( 0, (int)(node.getBaseType()) ) ;
				ASSERT_EQ  ( PlacementModifier::NONE, node.getPlacementModifier() ) ;
				ASSERT_EQ  ( 0, node.getBoundaryDefinitionIndex() ) ;
			}

	tileLayout.saveToVolFile(testOutDirectory + "tiles_noTiles") ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, generateTiledLayout_singleTile )
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;
	NodeLayout nodeLayoutTiled = tileLayout.generateLayoutWithMarkedTiles() ;

	ASSERT_EQ( nodeLayoutTiled.getNodeType(1,1,1), NodeBaseType::FLUID ) ;
	ASSERT_EQ( nodeLayoutTiled.getNodeType(2,2,2), NodeBaseType::FLUID ) ;

	ASSERT_EQ( nodeLayoutTiled.getNodeType(0,0,0), NodeBaseType::MARKER ) ;
	ASSERT_EQ( nodeLayoutTiled.getNodeType(0,0,3), NodeBaseType::MARKER ) ;
	ASSERT_EQ( nodeLayoutTiled.getNodeType(0,3,0), NodeBaseType::MARKER ) ;
	ASSERT_EQ( nodeLayoutTiled.getNodeType(0,3,3), NodeBaseType::MARKER ) ;
	ASSERT_EQ( nodeLayoutTiled.getNodeType(3,0,0), NodeBaseType::MARKER ) ;
	ASSERT_EQ( nodeLayoutTiled.getNodeType(3,0,3), NodeBaseType::MARKER ) ;
	ASSERT_EQ( nodeLayoutTiled.getNodeType(3,3,0), NodeBaseType::MARKER ) ;
	ASSERT_EQ( nodeLayoutTiled.getNodeType(3,3,3), NodeBaseType::MARKER ) ;

	for (auto it = nodeLayoutTiled.begin() ; it < nodeLayoutTiled.end() ; it++)
	{
		it->setBoundaryDefinitionIndex(0) ;
	}	
	
	tileLayout.saveToVolFile(testOutDirectory + "tiles_singleTile") ;

	std::cout << tileLayout.computeTilingStatistic().computeStatistics() << "\n" ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, pngImage )
{
	NodeLayout nodeLayout(
												getDefaultPixelClassificator(),
												Image("./test_data/geometries/div_1.png"), 
												10
											 ) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	tileLayout.saveToVolFile(testOutDirectory + "div_1_layout") ;

	std::cout << tileLayout.computeTilingStatistic().computeStatistics() << "\n" ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, pngImage2 )
{
	NodeLayout nodeLayout(
												getDefaultPixelClassificator(),
												Image("./test_data/geometries/div_2.png"), 
												10
											 ) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	tileLayout.saveToVolFile(testOutDirectory + "div_2_layout") ;

	std::cout << tileLayout.computeTilingStatistic().computeStatistics() << "\n" ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, DISABLED_largePngImage )
//TEST( TileLayout, largePngImage )
{
	NodeLayout nodeLayout(
												getDefaultPixelClassificator(),
												Image("./test_data/geometries/pre_A3_5000x4000.png"), 
												1
											 ) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	tileLayout.saveToVolFile(testOutDirectory + "pre_A3_5000x4000_layout") ;
	
	testCopyTileLayoutGPU( tileLayout ) ;
}



TEST( TileLayout, getBeginOfNonEmptyTiles_noThrow_0_Tiles )
{
	TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout() ;

	ASSERT_NO_THROW( tileLayout.getBeginOfNonEmptyTiles() ) ;
}



TEST( TileLayout, getBeginOfNonEmptyTiles_noThrow_1_Tile )
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout() ;

	ASSERT_NO_THROW( tileLayout.getBeginOfNonEmptyTiles() ) ;
}



TEST( TileLayout, getEndOfNonEmptyTiles_noThrow_0_Tiles )
{
	TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout() ;

	ASSERT_NO_THROW( tileLayout.getEndOfNonEmptyTiles() ) ;
}



TEST( TileLayout, getEndOfNonEmptyTiles_noThrow_1_Tile )
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout() ;

	ASSERT_NO_THROW( tileLayout.getEndOfNonEmptyTiles() ) ;
}



void testLoopIterator( TileLayout<StorageOnCPU> & tileLayout, size_t expectedCounter )
{
	size_t counter = 0 ;

	ASSERT_NO_THROW
	( 
			for (auto it = tileLayout.getBeginOfNonEmptyTiles() ;
								it < tileLayout.getEndOfNonEmptyTiles() ;
								it ++ )
			{
				counter ++ ;
			}
	) ;

	EXPECT_EQ( expectedCounter, counter ) ;
}



TEST( TileLayout, loopIteratorNoTiles )
{
	TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout() ;

	testLoopIterator( tileLayout, 0 ) ;
}



TEST( TileLayout, loopIteratorSingleTile )
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout() ;

	testLoopIterator( tileLayout, 1 ) ;
}



TEST( TileLayout, loopIteratorTwoTiles )
{
	TileLayout<StorageOnCPU> tileLayout = generateTwoTilesLayout() ;

	testLoopIterator( tileLayout, 2 ) ;
}



TEST( TileLayout, getTile_noTiles )
{
	TileLayout<StorageOnCPU> tileLayout = generateNoTilesLayout() ;
	auto wrongIterator = tileLayout.getBeginOfNonEmptyTiles() ;

	ASSERT_DEATH( tileLayout.getTile( wrongIterator ), "" ) ;
}



TEST( TileLayout, getTile )
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout() ;
	auto iterator = tileLayout.getBeginOfNonEmptyTiles() ;

	TileLayout<StorageOnCPU>::NonEmptyTile tile = tileLayout.getTile( iterator )  ;

	EXPECT_EQ( Coordinates(0,0,0), tile.getCornerPosition() ) ;
}



TEST( TileLayout, unpack )
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout() ;
	TileLayout<StorageOnCPU>::NonEmptyTile tile = tileLayout.getTile( tileLayout.getBeginOfNonEmptyTiles() )  ;

	for (size_t z=0 ; z < DEFAULT_3D_TILE_EDGE ; z++)
		for (size_t y=0 ; y < DEFAULT_3D_TILE_EDGE ; y++)
			for (size_t x=0 ; x < DEFAULT_3D_TILE_EDGE ; x++)
			{
				size_t linearIndex = x + y * DEFAULT_3D_TILE_EDGE + 
									z * DEFAULT_3D_TILE_EDGE * DEFAULT_3D_TILE_EDGE ;

				Coordinates c = tile.unpack( linearIndex ) ;

				ASSERT_EQ ( x, c.getX() ) ;
				ASSERT_EQ ( y, c.getY() ) ;
				ASSERT_EQ ( z, c.getZ() ) ;
			}
}



TEST( TileLayout, hasNeighbour_singleTile )
{
	const unsigned tileEdge = TileLayout<StorageOnCPU>::NonEmptyTile::getNNodesPerEdge() ;

	unsigned width  = tileEdge * 3 ;
	unsigned height = tileEdge * 3 ;
	unsigned depth  = tileEdge * 3 ;

	NodeLayout nodeLayout = createSolidNodeLayout( width, height, depth) ;

	nodeLayout.setNodeType( tileEdge, tileEdge, tileEdge, NodeBaseType::FLUID) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	EXPECT_EQ( tileLayout.getSize(), Size(3,3,3) ) ;
	EXPECT_EQ( tileLayout.getNoNonEmptyTiles(), 1u ) ;

	TileLayout<StorageOnCPU>::NonEmptyTile tile = tileLayout.getTile( 
			tileLayout.getBeginOfNonEmptyTiles()
			) ;
	EXPECT_EQ( tile.getCornerPosition(), Coordinates(tileEdge, tileEdge, tileEdge) ) ;
	EXPECT_EQ( tile.getMapPosition(), Coordinates(1, 1, 1) ) ;

	for (auto direction : Direction::D3Q19 )
	{
		EXPECT_FALSE( tileLayout.hasNeighbour( tile, direction) ) ;
	}

	EXPECT_ANY_THROW( tileLayout.getNeighbour(tile, Direction::TOP) ) ;
}



TEST( TileLayout, hasNeighbour_twoTiles )
{
	const unsigned tileEdge = TileLayout<StorageOnCPU>::NonEmptyTile::getNNodesPerEdge() ;

	unsigned width  = tileEdge * 3 ;
	unsigned height = tileEdge * 3 ;
	unsigned depth  = tileEdge * 3 ;

	NodeLayout nodeLayout = createSolidNodeLayout( width, height, depth) ;

	nodeLayout.setNodeType( tileEdge, tileEdge, tileEdge, NodeBaseType::FLUID) ;
	// Top
	nodeLayout.setNodeType( tileEdge, tileEdge, 2*tileEdge, NodeBaseType::FLUID) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	EXPECT_EQ( tileLayout.getSize(), Size(3,3,3) ) ;
	EXPECT_EQ( tileLayout.getNoNonEmptyTiles(), 2u ) ;

	TileLayout<StorageOnCPU>::NonEmptyTile tile = tileLayout.getTile( 
			tileLayout.getBeginOfNonEmptyTiles()
			) ;
	EXPECT_EQ( tile.getCornerPosition(), Coordinates(tileEdge, tileEdge, tileEdge) ) ;
	EXPECT_EQ( tile.getMapPosition(), Coordinates(1, 1, 1) ) ;

	EXPECT_TRUE( tileLayout.hasNeighbour( tile, Direction::TOP ) ) ;

	for (auto direction : Direction::D3Q19 )
	{
		if ( Direction::TOP != direction )
		{
			EXPECT_FALSE( tileLayout.hasNeighbour( tile, direction) ) ;
		}
	}

	EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::TOP) ) ;
	TileLayout<StorageOnCPU>::NonEmptyTile topTile = tileLayout.getNeighbour(tile, Direction::TOP) ;

	EXPECT_EQ( topTile.getCornerPosition(), Coordinates(tileEdge, tileEdge, 2*tileEdge) ) ;
	EXPECT_EQ( topTile.getMapPosition(), Coordinates(1, 1, 2) ) ;
}



TEST( TileLayout, hasNeighbour_allTiles )
{
	const unsigned tileEdge = TileLayout<StorageOnCPU>::NonEmptyTile::getNNodesPerEdge() ;

	unsigned width  = tileEdge * 3 ;
	unsigned height = tileEdge * 3 ;
	unsigned depth  = tileEdge * 3 ;

	NodeLayout nodeLayout = createFluidNodeLayout( width, height, depth) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	EXPECT_EQ( tileLayout.getSize(), Size(3,3,3) ) ;
	EXPECT_EQ( tileLayout.getNoNonEmptyTiles(), 27u ) ;

	TileLayout<StorageOnCPU>::NonEmptyTile tile = tileLayout.getTile( 13 	) ;
	EXPECT_EQ( tile.getCornerPosition(), Coordinates(tileEdge, tileEdge, tileEdge) ) ;
	EXPECT_EQ( tile.getMapPosition(), Coordinates(1, 1, 1) ) ;

	for (auto direction : Direction::D3Q19 )
	{
		EXPECT_TRUE( tileLayout.hasNeighbour( tile, direction) ) ;
	}

	/*
		Check neighbours
	*/
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::TOP) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::TOP) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(tileEdge, tileEdge, 2*tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(1, 1, 2) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::BOTTOM) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::BOTTOM) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(tileEdge, tileEdge, 0) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(1, 1, 0) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::EAST) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::EAST) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(2*tileEdge, tileEdge, tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(2, 1, 1) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::WEST) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::WEST) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(0, tileEdge, tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(0, 1, 1) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::SOUTH) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::SOUTH) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(tileEdge, 0, tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(1, 0, 1) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::NORTH) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::NORTH) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(tileEdge, 2*tileEdge, tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(1, 2, 1) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::NORTH + Direction::EAST) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::NORTH +
																																	 Direction::EAST) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(2*tileEdge, 2*tileEdge, tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(2, 2, 1) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::NORTH + Direction::WEST) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::NORTH +
																																	 Direction::WEST) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(0, 2*tileEdge, tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(0, 2, 1) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::SOUTH + Direction::EAST) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::SOUTH +
																																	 Direction::EAST) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(2*tileEdge, 0, tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(2, 0, 1) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::SOUTH + Direction::WEST) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::SOUTH +
																																	 Direction::WEST) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(0, 0, tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(0, 0, 1) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::SOUTH + Direction::TOP) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::SOUTH +
																																	 Direction::TOP) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(tileEdge, 0, 2*tileEdge) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(1, 0, 2) ) ;
	}
	{
		EXPECT_NO_THROW( tileLayout.getNeighbour(tile, Direction::SOUTH + Direction::BOTTOM) ) ;
		TileLayout<StorageOnCPU>::NonEmptyTile tile2 = tileLayout.getNeighbour(tile, Direction::SOUTH +
																																	 Direction::BOTTOM) ;

		EXPECT_EQ( tile2.getCornerPosition(), Coordinates(tileEdge, 0, 0) ) ;
		EXPECT_EQ( tile2.getMapPosition(), Coordinates(1, 0, 0) ) ;
	}
}



void testCornerTile( Coordinates corner )
{
	const unsigned tileEdge = TileLayout<StorageOnCPU>::NonEmptyTile::getNNodesPerEdge() ;

	unsigned width  = tileEdge * 3 ;
	unsigned height = tileEdge * 3 ;
	unsigned depth  = tileEdge * 3 ;

	NodeLayout nodeLayout = createSolidNodeLayout( width, height, depth) ;

	Coordinates cornerCoordinates(
														corner.getX() * tileEdge,
														corner.getY() * tileEdge,
														corner.getZ() * tileEdge
													) ;

	nodeLayout.setNodeType( cornerCoordinates, NodeBaseType::FLUID) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	EXPECT_EQ( tileLayout.getSize(), Size(3,3,3) ) ;
	EXPECT_EQ( tileLayout.getNoNonEmptyTiles(), 1u ) ;

	TileLayout<StorageOnCPU>::Iterator iterator = tileLayout.getBeginOfNonEmptyTiles() ;
	TileLayout<StorageOnCPU>::NonEmptyTile tile = tileLayout.getTile( iterator ) ;

	EXPECT_EQ( tile.getCornerPosition(), cornerCoordinates ) ;
	EXPECT_EQ( tile.getMapPosition(), corner ) ;

	for (auto direction : Direction::D3Q19 )
	{
		EXPECT_FALSE( tileLayout.hasNeighbour( tile, direction) ) ;
	}
}



TEST( TileLayout, hasNeighbour_corners )
{
	testCornerTile( Coordinates(0,0,0) ) ;
	testCornerTile( Coordinates(2,0,0) ) ;
	testCornerTile( Coordinates(0,2,0) ) ;
	testCornerTile( Coordinates(2,2,0) ) ;

	testCornerTile( Coordinates(0,0,2) ) ;
	testCornerTile( Coordinates(2,0,2) ) ;
	testCornerTile( Coordinates(0,2,2) ) ;
	testCornerTile( Coordinates(2,2,2) ) ;
}



__global__ void
kernelCheckTileLayout( TileLayout<StorageInKernel> tileLayout )
{
	if (0 == threadIdx.x  &&  0 == threadIdx.y  &&  0 == threadIdx.z)
	{
		/*
			Size test
		*/
		{
		Size s = tileLayout.getSize() ;
		printf("KERNEL tileLayout.size = (%lu, %lu, %lu)\n", 
						s.getWidth(), s.getHeight(), s.getDepth() ) ;
		}
		{
		Size s(5,6,7) ;
		printf("KERNEL tileLayout.size = (%lu, %lu, %lu)\n", 
						s.getWidth(), s.getHeight(), s.getDepth() ) ;
		}

		if ( Size(5,6,7) != tileLayout.getSize() )
			THROW ("Size differs") ;


		/*
			Vector (StorageInKernel) size test
		*/
		printf("KERNEL getNoNonEmptyTiles = %lu\n", tileLayout.getNoNonEmptyTiles() ) ;

		if ( 5*6*7 != tileLayout.getNoNonEmptyTiles() )
			THROW ("getNoNonEmptyTiles differs") ;


		/*
			Vectors (StorageInKernel) content and getCornerPosition() test
		*/
		for (auto it = tileLayout.getBeginOfNonEmptyTiles() ;
							it < tileLayout.getEndOfNonEmptyTiles() ; 
							it++ )
		{
			auto tile = tileLayout.getTile( it ) ;
			auto corner = tile.getCornerPosition() ;
			printf("corner: (%lu,%lu,%lu) ", corner.getX(), corner.getY(), corner.getZ() ) ;
			auto map = tile.getMapPosition() ;
			printf("map: (%lu,%lu,%lu) ", map.getX(), map.getY(), map.getZ() ) ;
			printf("\t\t") ;
		}
		printf("\n") ;

		unsigned i=0 ;
		for (unsigned x=0 ; x < 5 ; x++)
			for (unsigned y=0 ; y < 6 ; y++)
				for (unsigned z=0 ; z < 7 ; z++)
				{
					auto tile = tileLayout.getTile( i ) ;

					Coordinates s = tile.getCornerPosition() ;
					auto m = tile.getMapPosition() ;

					if ( Coordinates(x*4,y*4,z*4) != s )
						THROW ("getCornerPosition differs") ;

					if ( Coordinates(x,y,z) != m )
						THROW ("getMapPosition differs") ;

					i++ ;
				}

		/*
			hasNeighbour() and getNeighbour() tests
		*/
		{
			auto tile = tileLayout.getTile(0) ; 

			for (auto d : { N, E, T, NE, NT, ET, NET } )
			{
				if ( not tileLayout.hasNeighbour( tile, d ) )
				{
					printf("Tile 0 has  **NOT** %s neighbor\n", toString(d) ) ;
					THROW ("") ;
				}

				auto neighborTile = tileLayout.getNeighbour( tile, d ) ;
				printf("Index of neighbor %s of tile 0 is %u\n", toString(d), neighborTile.getIndex() ) ;
			}
			for (auto d : { S, W, B, SW, SB, SE, ST, WB, WT, SWT, SWB, SET, SWT, NEB, NWB, SEB, NWT, NW, NB, EB } )
			{
				if ( tileLayout.hasNeighbour( tile, d ) )
				{
					printf("Tile 0 **HAS** %s neighbor\n", toString(d) ) ;
					THROW ("") ;
				}
			}
		}

		{
			auto tile = tileLayout.getTile(50) ; // somewere in the middle

			for (auto d : { N,S,E,T,W,B, NE,NW,NT,NB, SE,SW,ST,SB, ET,EB,WT,WB, NET,NWT,SET,SWT,NEB,NWB,SEB,SWB } )
			{
				if ( not tileLayout.hasNeighbour( tile, d ) )
				{
					printf("Tile 50 has  **NOT** %s neighbor\n", toString(d) ) ;
					THROW ("") ;
				}
			}
		}

	}
}



TEST( TileLayoutKernel, constructor )
{
	const unsigned tileEdge = 4 ;
	unsigned width  = tileEdge * 5 ;
	unsigned height = tileEdge * 6 ;
	unsigned depth  = tileEdge * 7 ;

	NodeLayout nodeLayout = createFluidNodeLayout( width, height, depth) ;
	TileLayout<StorageOnCPU> tileLayoutCPU( nodeLayout ) ;
	
	{
		Size s = tileLayoutCPU.getSize() ;
		printf("CPU tileLayout.size = (%lu, %lu, %lu)\n", 
						s.getWidth(), s.getHeight(), s.getDepth() ) ;
	}
	printf("CPU getNoNonEmptyTiles = %lu\n", tileLayoutCPU.getNoNonEmptyTiles() ) ;


	TileLayout<StorageOnGPU> tileLayoutGPU( tileLayoutCPU ) ;
	TileLayout<StorageInKernel> tileLayoutKernel( tileLayoutGPU ) ;

	dim3 numBlocks(1) ;
	dim3 numThreads(1) ;

	EXPECT_NO_THROW
	(
		( kernelCheckTileLayout<<<numBlocks, numThreads>>>( tileLayoutKernel ) ) ;
		CUDA_CHECK( cudaPeekAtLastError() );
		CUDA_CHECK( cudaDeviceSynchronize() );  	
	) ;
}



void buildTileLayoutForCase (const string caseName)
{
		string casePath = "./test_data/cases/" + caseName ;

		logger << "Reading simulation from " << casePath << "\n" ;

		unique_ptr <Settings> settings ;
		PerformanceMeter pm ;

		ASSERT_NO_THROW (settings.reset (new Settings (casePath))) ;

		ColoredPixelClassificator 
			coloredPixelClassificator (settings->getPixelColorDefinitionsFilePath()) ;

		pm.start() ;
		Image image (settings->getGeometryPngImagePath()) ;
		pm.stop() ;
		logger << "Image read: " << pm.generateSummary() << "\n" ;
		pm.clear() ;

		pm.start() ;
		NodeLayout nodeLayout (coloredPixelClassificator, image, 
													 settings->getZExpandDepth()) ;
		pm.stop() ;
		logger << "NodeLayout: " << pm.generateSummary() << "\n" ;
		pm.clear() ;

		pm.start() ;
		settings->initialModify (nodeLayout) ;
		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		if ( 1 < settings->getZExpandDepth() )
		{
			expandedNodeLayout.computeSolidNeighborMasks() ;
			expandedNodeLayout.computeNormalVectors() ;
			nodeLayout.temporaryMarkUndefinedBoundaryNodesAndCovers() ;
			nodeLayout.restoreBoundaryNodes (coloredPixelClassificator, image) ;
			expandedNodeLayout.classifyNodesPlacedOnBoundary (*settings) ;
			expandedNodeLayout.classifyPlacementForBoundaryNodes (*settings) ;
		}
		settings->finalModify (nodeLayout) ;
		//TODO: unoptimal (computed twice), but easy.
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;
		pm.stop() ;
		logger << "ExpandedNodeLayout: " << pm.generateSummary() << "\n" ;
		pm.clear() ;

	ASSERT_NO_THROW
	(
		pm.start() ;
		TileLayout <StorageOnCPU> tileLayout (nodeLayout) ;
		pm.stop() ;
		logger << "TileLayout building: " << pm.generateSummary() << "\n" ;
		pm.clear() ;
	) ;
}



TEST (TileLayout, build_A2_1000x1000)
{
	buildTileLayoutForCase ("A2_1000x1000") ;
}



TEST (TileLayout, build_A2_2200x2200_GPU)
{
	buildTileLayoutForCase ("A2_2200x2200_GPU") ;
}



TEST (TileLayout, build_A3_5000x4000_GPU)
{
	buildTileLayoutForCase ("A3_5000x4000_GPU") ;
}


