#include "gtest/gtest.h"


#include "Tile.hpp"
#include "Storage.hpp"

#include "TileTest.hpp"
#include "TiledLatticeTest.hpp"



using namespace microflow ;
using namespace std ;



TEST( Tile, getNNodesPerTile )
{
	EXPECT_EQ( (Tile< D3Q19, double, 4, StorageOnCPU, TileDataArrangement::XYZ  >::getNNodesPerTile()), 64u ) ;
	EXPECT_EQ( (Tile< D3Q19, double, 4, StorageOnCPU, TileDataArrangement::OPT_1>::getNNodesPerTile()), 64u ) ;
	EXPECT_EQ( (Tile< D3Q27, double, 4, StorageOnCPU, TileDataArrangement::XYZ  >::getNNodesPerTile()), 64u ) ;
	EXPECT_EQ( (Tile< D3Q27, double, 4, StorageOnCPU, TileDataArrangement::OPT_1>::getNNodesPerTile()), 64u ) ;
}



TEST( Tile, getNValuesPerNode )
{
	EXPECT_EQ( (Tile< D3Q19, double, 4, StorageOnCPU, TileDataArrangement::XYZ  >::getNValuesPerNode()), 2u*19u + 3u*(3u+1u) ) ;
	EXPECT_EQ( (Tile< D3Q19, double, 4, StorageOnCPU, TileDataArrangement::OPT_1>::getNValuesPerNode()), 2u*19u + 3u*(3u+1u) ) ;

	EXPECT_EQ( (Tile< D3Q27, double, 4, StorageOnCPU, TileDataArrangement::XYZ  >::getNValuesPerNode()), 2u*27u + 3u*(3u+1u) ) ;
	EXPECT_EQ( (Tile< D3Q27, double, 4, StorageOnCPU, TileDataArrangement::OPT_1>::getNValuesPerNode()), 2u*27u + 3u*(3u+1u) ) ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTileGetFPtrCPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	TLattice tiledLattice( tileLayout, expandedNodeLayout, Settings() ) ;
	auto tile = tiledLattice.getTile(0) ;

	DataType * f[LatticeArrangement::getQ()] ;
	
	for (unsigned i=0 ; i < LatticeArrangement::getQ() ; i++)
	{
		EXPECT_NO_THROW( f[i] = tile.getFPtr( LatticeArrangement::c[i] ) ) ;
		std::cout << "f" << i << "_ptr = " << f[i] << "\n" ;
	}
	
	//TODO: replace 64 with tile.getNNodesPerTile() ?
	for (unsigned i=0 ; i < LatticeArrangement::getQ() ; i++)
	{
		EXPECT_EQ( i*64, f[i] - f[0] ) << " for i = " << i ;
	}
}



TEST( TileCPU_XYZ, getFPtr)
{
	testTileGetFPtrCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST( TileCPU_OPT_1, getFPtr)
{
	testTileGetFPtrCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTileGetFPostPtrCPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	TLattice tiledLattice( tileLayout, expandedNodeLayout, Settings() ) ;
	auto tile = tiledLattice.getTile(0) ;

	DataType * f[LatticeArrangement::getQ()] ;
	
	for (unsigned i=0 ; i < LatticeArrangement::getQ() ; i++)
	{
		EXPECT_NO_THROW( f[i] = tile.getFPostPtr( LatticeArrangement::c[i] ) ) ;
		std::cout << "f" << i << "_ptr = " << f[i] << "\n" ;
	}

	EXPECT_EQ( f[0], tile.getFPtr( LatticeArrangement::c[LatticeArrangement::getQ()-1] )  +
									 tile.getNNodesPerTile() ) ;
	
	//TODO: replace 64 with tile.getNNodesPerTile() ?
	for (unsigned i=0 ; i < LatticeArrangement::getQ() ; i++)
	{
		EXPECT_EQ( i*64, f[i] - f[0] ) << " for i = " << i ;
	}
}



TEST( TileCPU_XYZ, getFPostPtr)
{
	testTileGetFPostPtrCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST( TileCPU_OPT_1, getFPostPtr)
{
	testTileGetFPostPtrCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTileNoOverlapValuesSingleTileCPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	TLattice tiledLattice( tileLayout, expandedNodeLayout, Settings() ) ;

	EXPECT_EQ( tiledLattice.getNOfTiles(), 1u ) ;

	auto tile = tiledLattice.getTile(0) ;

	unsigned val = 0 ;
	fillTile (tile, val) ;
	val = 0 ;
	checkTile (tile, val) ;
	val = 0 ;
	checkTileNodes( tile, val ) ;
}



TEST (TileCPU_XYZ, noOverlapValuesSingleTile)
{
	testTileNoOverlapValuesSingleTileCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST (TileCPU_OPT_1, noOverlapValuesSingleTile)
{
	testTileNoOverlapValuesSingleTileCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTileNoOverlapValuesTwoTilesCPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateTwoTilesLayout() ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	TLattice tiledLattice( tileLayout, expandedNodeLayout, Settings() ) ;	

	EXPECT_EQ (tiledLattice.getNOfTiles(), 2u) ;

	unsigned val = 0 ;
	for (unsigned t=0 ; t < 2 ; t++)
	{
		auto tile = tiledLattice.getTile(t) ;
		fillTile (tile, val) ;
	}
	val = 0 ;
	for (unsigned t=0 ; t < 2 ; t++)
	{
		auto tile = tiledLattice.getTile(t) ;
		checkTile (tile, val) ;
	}
	val = 0 ;
	for (unsigned t=0 ; t < 2 ; t++)
	{
		auto tile = tiledLattice.getTile(t) ;
		checkTileNodes (tile, val) ;
	}
}



TEST (TileCPU_XYZ, noOverlapValuesTwoTiles)
{
	testTileNoOverlapValuesTwoTilesCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST (TileCPU_OPT_1, noOverlapValuesTwoTiles)
{
	testTileNoOverlapValuesTwoTilesCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTileVelocityArrayCPU()
{
	unsigned val = 0 ;

	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;
	auto tile = tiledLattice.getTile(0) ;

	const unsigned edge = tile.getNNodesPerEdge() ;

	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				tile.u( Axis::X )[z][y][x] = ++val ; 
				tile.u( Axis::Y )[z][y][x] = ++val ; 
				tile.u( Axis::Z )[z][y][x] = ++val ; 
				tile.uT0( Axis::X )[z][y][x] = ++val ; 
				tile.uT0( Axis::Y )[z][y][x] = ++val ; 
				tile.uT0( Axis::Z )[z][y][x] = ++val ; 
				tile.uBoundary( Axis::X )[z][y][x] = ++val ; 
				tile.uBoundary( Axis::Y )[z][y][x] = ++val ; 
				tile.uBoundary( Axis::Z )[z][y][x] = ++val ; 

			}

	val = 0 ;
	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				ASSERT_EQ( tile.u()[0][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.u()[1][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.u()[2][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

				ASSERT_EQ( tile.uT0()[0][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.uT0()[1][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.uT0()[2][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

				ASSERT_EQ( tile.uBoundary()[0][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.uBoundary()[1][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.uBoundary()[2][z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
			}
}



TEST (TileCPU_XYZ, velocityArray)
{
	testTileVelocityArrayCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST (TileCPU_OPT_1, velocityArray)
{
	testTileVelocityArrayCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



static inline
void print (double (&a) [4][4][4], unsigned offset = 0)
{
	const unsigned edge = 4 ;

	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				cout << setw(2) << a [z][y][x] - offset << " " ;
			}
}



template <DataStorageMethod DataStorage>
static inline
void testNodeFromTileAccess ()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout() ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice< D3Q19, double, StorageOnCPU, TileDataArrangement::OPT_1 >  TLattice ;

	TLattice tiledLattice( tileLayout, expandedNodeLayout, Settings() ) ;	

	EXPECT_EQ( tiledLattice.getNOfTiles(), 1u ) ;

	const unsigned tileIndex = 0 ;
	auto tile = tiledLattice.getTile (tileIndex) ;

	const unsigned edge = TLattice::TileType::TraitsType::getNNodesPerEdge() ;

	const unsigned nodesPerTile = edge * edge * edge ;
	unsigned v0  = 0 ;
	unsigned vE  = 1  * nodesPerTile ;
	unsigned vW  = 2  * nodesPerTile ;
	unsigned vNE = 3  * nodesPerTile ;
	unsigned vSE = 4  * nodesPerTile ;
	unsigned vET = 5  * nodesPerTile ;
	unsigned vEB = 6  * nodesPerTile ;
	unsigned vNW = 7  * nodesPerTile ;
	unsigned vSW = 8  * nodesPerTile ;
	unsigned vWT = 9  * nodesPerTile ;
	unsigned vWB = 10 * nodesPerTile ;

	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				auto node = tile.getNode<DataStorage> (x,y,z) ;

				node.f (O) = v0 ;
				node.f (E) = vE ;
				node.f (W) = vW ;

				node.f (NE) = vNE ;
				node.f (SE) = vSE ;
				node.f (ET) = vET ;
				node.f (EB) = vEB ;
				node.f (NW) = vNW ;
				node.f (SW) = vSW ;
				node.f (WT) = vWT ;
				node.f (WB) = vWB ;

				v0++ ;
				vE++ ;
				vW++ ;
				vNE++ ;
				vSE++ ;
				vET++ ;
				vEB++ ;
				vNW++ ;
				vSW++ ;
				vWT++ ;
				vWB++ ;
			}

	auto fOArray = tile.f (O) ;
	auto fEArray = tile.f (E) ;

#define PRINT_F(direction, offset)    \
	cout << "f(" #direction ") : " ;    \
	print (tile.f(direction), offset) ; \
	cout << "\n" ;

	PRINT_F (O, 0) ;
	PRINT_F (E, 1u * nodesPerTile) ;
	PRINT_F (W, 2u * nodesPerTile) ;
	PRINT_F (NE, 3u * nodesPerTile) ;
	PRINT_F (SE, 4u * nodesPerTile) ;
	PRINT_F (ET, 5u * nodesPerTile) ;
	PRINT_F (EB, 6u * nodesPerTile) ;
	PRINT_F (NW, 7u * nodesPerTile) ;
	PRINT_F (SW, 8u * nodesPerTile) ;
	PRINT_F (WT, 9u * nodesPerTile) ;
	PRINT_F (WB, 10u * nodesPerTile) ;

#undef PRINT_F

	unsigned vtst = 0 ;

	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				EXPECT_EQ (vtst, tile.f(O)[z][y][x]) 
					<< "at x=" << x << ", y=" << y << ", z=" << z << "\n" ;
				vtst++ ;
			}


#define TEST_ZXY(direction)                                           \
	for (unsigned z=0 ; z < edge ; z++)                                 \
		for (unsigned x=0 ; x < edge ; x++)                               \
			for (unsigned y=0 ; y < edge ; y++)                             \
			{                                                               \
				EXPECT_EQ (vtst, tile.f(direction)[z][y][x])                  \
					<< "at x=" << x << ", y=" << y << ", z=" << z << "\n" ;     \
				vtst++ ;                                                      \
			}

	TEST_ZXY (E) ;
	TEST_ZXY (W) ;
	vtst += 64 ;
	//TEST_ZXY (NE) ;
	vtst += 64 ;
	//TEST_ZXY (SE) ;
	TEST_ZXY (ET) ;
	TEST_ZXY (EB) ;
	TEST_ZXY (NW) ;
	TEST_ZXY (SW) ;
	TEST_ZXY (WT) ;
	TEST_ZXY (WB) ;

#undef TEST_ZXY
}


TEST (Tile_OPT_1, nodeFromTileAccess_REFERENCE)
{
	testNodeFromTileAccess <DataStorageMethod::REFERENCE> () ;
}


TEST (Tile_OPT_1, nodeFromTileAccess_POINTERS)
{
	testNodeFromTileAccess <DataStorageMethod::POINTERS> () ;
}

