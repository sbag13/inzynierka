#ifndef TILE_LAYOUT_TEST_HPP
#define TILE_LAYOUT_TEST_HPP



#include "gtest/gtest.h"

#include "TileLayout.hpp"
#include "TilingStatistic.hpp"

#include "NodeLayoutTest.hpp"
#include "ColoredPixelClassificatorTest.hpp"
#include "TileDefinitions.hpp"



using namespace microflow ;



inline
TileLayout<StorageOnCPU> generateNoTilesLayout(const size_t width, const size_t height, const size_t depth)
{
	NodeLayout nodeLayout = createSolidNodeLayout(width, height, depth) ;
	return TileLayout<StorageOnCPU> ( nodeLayout ) ;
}



inline
TileLayout<StorageOnCPU> generateNoTilesLayout()
{
	return generateNoTilesLayout(21, 121, 11) ;
}



inline
TileLayout<StorageOnCPU> generateSingleTileLayout(const size_t width, const size_t height, const size_t depth)
{
	EXPECT_TRUE( width > 1  ) ;
	EXPECT_TRUE( height > 2 ) ;

	NodeLayout nodeLayout = createSolidNodeLayout( width, height, depth) ;

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType(2, 2, 2, NodeBaseType::FLUID) ;
	
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	return tileLayout ;
}



inline
TileLayout<StorageOnCPU> generateSingleTileLayout()
{
	return
	generateSingleTileLayout( DEFAULT_3D_TILE_EDGE, DEFAULT_3D_TILE_EDGE, DEFAULT_3D_TILE_EDGE ) ;
}



inline
TileLayout<StorageOnCPU> generateTwoTilesLayout(const size_t width, const size_t height, const size_t depth)
{
	EXPECT_TRUE( width > 4 ) ;
	EXPECT_TRUE( height > 2 ) ;
	EXPECT_TRUE( depth > 2 ) ;

	NodeLayout nodeLayout = createSolidNodeLayout( width, height, depth ) ;

	nodeLayout.setNodeType(1, 1, 1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType(2, 2, 2, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType(4, 1, 1, NodeBaseType::FLUID) ;

	return TileLayout<StorageOnCPU>( nodeLayout ) ;
}



inline
TileLayout<StorageOnCPU> generateTwoTilesLayout()
{
	return generateTwoTilesLayout( 21, 121, 11 ) ;
}



#endif
