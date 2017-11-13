#include "gtest/gtest.h"



#include "NodeFromTile.hpp"

#include "TileTest.hpp"
#include "TiledLatticeTest.hpp"



using namespace microflow ;



//FIXME: Duplicated code from TileTest.cc: checkTileNodes(...).
template <class TiledLattice>
static inline
void checkTileNodesWithPointers( TiledLattice & tiledLattice, 
																 unsigned tileIndex, unsigned & val )
{
	typedef typename TiledLattice::TileType Tile ;
	typedef typename TiledLattice::LatticeArrangementType LArrangement ;

	const unsigned edge = Tile::getNNodesPerEdge() ;

	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				auto node = NodeFromTile <Tile,DataStorageMethod::POINTERS> 
										(
											x,y,z, tileIndex,
											tiledLattice.getNodeTypesPointer(),
											tiledLattice.getSolidNeighborMasksPointer(),
											tiledLattice.getNodeNormalsPointer(),
											tiledLattice.getAllValuesPointer()
										) ;

				for (unsigned i=0 ; i < LArrangement::getQ() ; i++)
				{
					Direction::D direction = LArrangement::c[i] ;
					Direction::DirectionIndex di = LArrangement::getIndex( direction ) ;
					
					++val ;

					ASSERT_EQ( node.f( direction ), val ) \
						<< "x=" << x <<", y=" << y << ", z=" << z << ", d=" << i \
						<< ", val=" << val << "\n" ; 
					ASSERT_EQ( node.f( Direction(direction) ), val ) \
						<< "x=" << x <<", y=" << y << ", z=" << z << ", d=" << i \
						<< ", val=" << val << "\n" ; 
					ASSERT_EQ( node.f( di ), val ) \
						<< "x=" << x <<", y=" << y << ", z=" << z << ", d=" << i \
						<< ", val=" << val << "\n" ; 

					++val ;

					ASSERT_EQ( node.fPost( direction ), val ) \
						<< "x=" << x <<", y=" << y << ", z=" << z << ", d=" << i \
						<< ", val=" << val << "\n" ; 
					ASSERT_EQ( node.fPost( Direction(direction) ), val ) \
						<< "x=" << x <<", y=" << y << ", z=" << z << ", d=" << i \
						<< ", val=" << val << "\n" ; 
					ASSERT_EQ( node.fPost( di ), val ) \
						<< "x=" << x <<", y=" << y << ", z=" << z << ", d=" << i \
						<< ", val=" << val << "\n" ; 

				}

				ASSERT_EQ( node.rho(), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

					++val ;
				/*ASSERT_EQ( node.rho0(), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; */
				ASSERT_EQ( node.rhoBoundary(), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

				ASSERT_EQ( node.u( Axis::X ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.u( Axis::Y ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.u( Axis::Z ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

					//val += 3 ; // Instead of the next three tests.

				ASSERT_EQ( node.uT0( X ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.uT0( Y ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.uT0( Z ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

				ASSERT_EQ( node.uBoundary( Axis::X ), ++val )
					<< "x=" << x <<", y=" << y << ", z=" << z
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.uBoundary( Axis::Y ), ++val )
					<< "x=" << x <<", y=" << y << ", z=" << z
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.uBoundary( Axis::Z ), ++val )
					<< "x=" << x <<", y=" << y << ", z=" << z
					<< ", val=" << val-1 << "\n" ; 
			}
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
static inline
void testNodeFromTilePointersAccessorsCPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;

	EXPECT_EQ (tiledLattice.getNOfTiles(), 1u) ;	

	auto tile = tiledLattice.getTile (0) ;

	unsigned val = 0 ;
	fillTile (tile, val) ;
	val = 0 ;
	checkTileNodes (tile, val) ;
	val = 0 ;
	checkTileNodesWithPointers (tiledLattice, tile.getCurrentTileIndex(), val) ;
}



TEST (NodeFromTileCPU_XYZ_POINTERS, accessors)
{
	testNodeFromTilePointersAccessorsCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST (NodeFromTileCPU_OPT_1_POINTERS, accessors)
{
	testNodeFromTilePointersAccessorsCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



template <TileDataArrangement DataArrangement>
bool shouldFBeEqual (Direction direction) ;



template <>
bool shouldFBeEqual <TileDataArrangement::XYZ> (Direction direction __attribute__((unused)))
{
	return true ;
}



template <>
bool shouldFBeEqual <TileDataArrangement::OPT_1> (Direction direction)
{
	switch (direction.get())
	{
		case E:
		case W:
		case NE:
		case SE:
		case ET:
		case EB:
		case NW:
		case SW:
		case WT:
		case WB:
			return false ;

		default: 
			return true ;
	} ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
static inline
void testNodeFromTileArrayVsNodeCPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(8, 8, 8) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;
	typedef typename TLattice::TileType Tile ;
	typedef typename TLattice::LatticeArrangementType LArrangement ;

	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;

	EXPECT_EQ (tiledLattice.getNOfTiles(), 1u) ;	

	auto tile = tiledLattice.getTile (0) ;

	unsigned val = 0 ;
	fillTile (tile, val) ;
	val = 0 ;

	for (unsigned q=0 ; q < LArrangement::getQ() ; q++)
	{
		Direction direction = LArrangement::c[q] ;

		bool areFEqual = true ;
		bool areFPostEqual = true ;
		const unsigned edge = Tile::getNNodesPerEdge() ;

		for (unsigned z=0 ; z < edge ; z++)
			for (unsigned y=0 ; y < edge ; y++)
				for (unsigned x=0 ; x < edge ; x++)
				{
					auto node = tile.getNode (x,y,z) ;

					if (node.f (direction) != tile.f (direction) [z][y][x])
					{
						areFEqual = false ;
					}
					if (node.fPost (direction) != tile.fPost (direction) [z][y][x])
					{
						areFPostEqual = false ;
					}
				}

		
		auto fEqStr     = areFEqual     ? "EQUAL" : "NOT EQUAL" ;
		auto fPostEqStr = areFPostEqual ? "EQUAL" : "NOT EQUAL" ;

		std::cout << "f     (" << direction << ") are " << fEqStr     << "\n" ;
		std::cout << "fPost (" << direction << ") are " << fPostEqStr << "\n" ;

		EXPECT_EQ (shouldFBeEqual <DataArrangement> (direction), areFEqual) ;
		EXPECT_EQ (shouldFBeEqual <DataArrangement> (direction), areFPostEqual) ;
	}
}



TEST (NodeFromTileCPU_XYZ, arrayVsNode)
{
	testNodeFromTileArrayVsNodeCPU <D3Q19,double,TileDataArrangement::XYZ> () ;
}



TEST (NodeFromTileCPU_OPT_1, arrayVsNode)
{
	testNodeFromTileArrayVsNodeCPU <D3Q19,double,TileDataArrangement::OPT_1> () ;
}




