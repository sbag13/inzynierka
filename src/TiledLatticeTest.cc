#include "gtest/gtest.h"

#include "TiledLatticeTest.hpp"
#include "Writer.hpp"
#include "TestPath.hpp"



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTiledLatticeSingleTileCPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	EXPECT_NO_THROW (TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings())) ;
}



TEST (TiledLatticeCPU_XYZ, singleTile)
{
	testTiledLatticeSingleTileCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST (TiledLatticeCPU_OPT_1, singleTile)
{
	testTiledLatticeSingleTileCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTiledLatticeSingleTileGPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> 
		TLatticeCPU ;
	typedef TiledLattice <LatticeArrangement, DataType, StorageOnGPU, DataArrangement> 
		TLatticeGPU ;

	TLatticeCPU tiledLatticeCPU (tileLayout, expandedNodeLayout, Settings()) ;

	TileLayout<StorageOnGPU> tileLayoutGPU (tileLayout) ;
	EXPECT_NO_THROW (TLatticeGPU tiledLattice (tiledLatticeCPU, tileLayoutGPU)) ;
}



TEST (TiledLatticeGPU_XYZ, singleTile)
{
	testTiledLatticeSingleTileGPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST (TiledLatticeGPU_OPT_1, singleTile)
{
	testTiledLatticeSingleTileGPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTiledLatticeGetTileCPU()
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;
	
	typename TLattice::Iterator ti ;

	EXPECT_NO_THROW (ti = tiledLattice.getBeginOfTiles()) ;

	EXPECT_NO_THROW  (tiledLattice.getTile (ti)) ;
	EXPECT_ANY_THROW (tiledLattice.getTile (++ti)) ;
}



TEST (TiledLatticeCPU_XYZ, getTile)
{
	testTiledLatticeGetTileCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST (TiledLatticeCPU_OPT_1, getTile)
{
	testTiledLatticeGetTileCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



TEST (TiledLatticeCPU, saveVtk)
{
	NodeLayout nodeLayout = createSolidNodeLayout( 4,4,4 ) ;
	nodeLayout.setNodeType(0, 0, 0, NodeBaseType::FLUID) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	ASSERT_EQ( 1u, tileLayout.getNoNonEmptyTiles() ) ;
	auto tile = tileLayout.getTile( tileLayout.getBeginOfNonEmptyTiles() ) ;
	Coordinates corner = tile.getCornerPosition() ;

	ASSERT_EQ( 0u, corner.getX() ) ;
	ASSERT_EQ( 0u, corner.getY() ) ;
	ASSERT_EQ( 0u, corner.getZ() ) ;

	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <D3Q19, double, StorageOnCPU, TileDataArrangement::XYZ> TLattice ;
	typedef Writer <D3Q19, double, TileDataArrangement::XYZ> TWriter ;

	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;

	ASSERT_EQ( 0u, tiledLattice.getTile(0).getCornerPosition().getX() ) ;
	ASSERT_EQ( 0u, tiledLattice.getTile(0).getCornerPosition().getY() ) ;
	ASSERT_EQ( 0u, tiledLattice.getTile(0).getCornerPosition().getZ() ) ;

	Settings settings("./test_data/cases/configuration_1") ;
	
	ASSERT_NO_THROW
	( 
		auto writer = TWriter (tiledLattice) ;
		writer.saveVtk (settings, testOutDirectory + "out.vti") ;
	);
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
void testTiledLatticeOperatorEqCPU()
{
	TileLayout<StorageOnCPU> tileLayout1 = generateSingleTileLayout(21, 121, 11) ;

	NodeLayout nodeLayout1 = tileLayout1.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout1 (nodeLayout1) ;
	expandedNodeLayout1.computeSolidNeighborMasks() ;
	expandedNodeLayout1.computeNormalVectors() ;

	typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> TLattice ;

	TLattice tiledLattice1 (tileLayout1, expandedNodeLayout1, Settings()) ;
	TLattice tiledLattice2 (tileLayout1, expandedNodeLayout1, Settings()) ;

	EXPECT_TRUE (tiledLattice1 == tiledLattice2) ;

	TileLayout<StorageOnCPU> tileLayout2 = generateTwoTilesLayout() ;

	NodeLayout nodeLayout2 = tileLayout2.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout2 (nodeLayout2) ;
	expandedNodeLayout2.computeSolidNeighborMasks() ;
	expandedNodeLayout2.computeNormalVectors() ;

	TLattice tiledLattice3 (tileLayout2, expandedNodeLayout2, Settings()) ;

	EXPECT_FALSE (tiledLattice1 == tiledLattice3) ;

	auto tile2 = tiledLattice2.getTile(0) ;
	auto node2 = tile2.getNode(1,0,1) ; // BEWARE: node2 uses reference to tile2.

	{
		auto r1 = tile2.rho()[1][0][1] ;
		auto r2 = node2.rho() ;
		EXPECT_EQ (r1, r2) ;
	}

	auto oldF = tile2.f(E)[1][0][1] ;
	auto oldFPost = tile2.fPost(E)[1][0][1] ;

	tile2.f(E)[1][0][1] = 5.0 ;
	tile2.fPost(E)[1][0][1] = 6.0 ;

	for (unsigned z=0 ; z < 4 ; z++)
		for (unsigned y=0 ; y < 4 ; y++)
			for (unsigned x=0 ; x < 4 ; x++)
			{
				std::cout << 
					tile2.computeNodeDataIndex (x,y,z,tile2.getCurrentTileIndex(), 
																			TLattice::TileType::Data::F, E)
									<< "\t" ;
			}
	std::cout << "\n" ;
	for (unsigned z=0 ; z < 4 ; z++)
		for (unsigned y=0 ; y < 4 ; y++)
			for (unsigned x=0 ; x < 4 ; x++)
			{
				std::cout << tile2.f(E)[z][y][x] << "\t" ;
			}
	std::cout << "\n" ;

	for (unsigned z=0 ; z < 4 ; z++)
		for (unsigned y=0 ; y < 4 ; y++)
			for (unsigned x=0 ; x < 4 ; x++)
			{
				std::cout << tile2.getNode(x,y,z).f(E) << "\t" ;
			}
	std::cout << "\n" ;
  
	if (TileDataArrangement::XYZ == DataArrangement)
	{
		EXPECT_EQ (tile2.f(E)[1][0][1], node2.f(E)) ;
		EXPECT_EQ (tile2.fPost(E)[1][0][1], node2.fPost(E)) ;
	}
	if (TileDataArrangement::OPT_1 == DataArrangement)
	{
		EXPECT_NE (tile2.f(E)[1][0][1], node2.f(E)) ;
		EXPECT_NE (tile2.fPost(E)[1][0][1], node2.fPost(E)) ;
	}

	tile2.f(E)[1][0][1] = oldF ;
	tile2.fPost(E)[1][0][1] = oldFPost ;

	EXPECT_TRUE (tiledLattice1 == tiledLattice2) ;


	double tmp = node2.rho() ;
	node2.rho() = 999.0 ;

	EXPECT_EQ (tile2.rho()[1][0][1], node2.rho()) ;
	EXPECT_FALSE (tiledLattice1 == tiledLattice2) ;

	node2.rho() = tmp ;
	EXPECT_TRUE (tiledLattice1 == tiledLattice2) ;

	NodeType tmpNode = node2.nodeType().getBaseType() ;
	node2.nodeType().setBaseType (NodeBaseType::MARKER) ;

	EXPECT_FALSE (tiledLattice1 == tiledLattice2) ;

	node2.nodeType().setBaseType (tmpNode.getBaseType()) ;

	EXPECT_TRUE (tiledLattice1 == tiledLattice2) ;
}



TEST ( TiledLatticeCPU_XYZ, operatorEq )
{
	testTiledLatticeOperatorEqCPU <D3Q19, double, TileDataArrangement::XYZ> () ;
}



TEST ( TiledLatticeCPU_OPT_1, operatorEq )
{
	testTiledLatticeOperatorEqCPU <D3Q19, double, TileDataArrangement::OPT_1> () ;
}



template <TileDataArrangement DataArrangement>
__global__ void
kernelTestTiledLatticeGPU
	(
		TiledLattice <D3Q19, double, StorageInKernel, DataArrangement> tiledLattice
	)
{
	auto tile = tiledLattice.getTile (blockIdx.x) ;
	auto node = tile.getNode (threadIdx.x, threadIdx.y, threadIdx.z) ;

	
	NodeType expectedNode (NodeBaseType::SOLID) ;
	if ( 0 == threadIdx.x  &&  0 == threadIdx.y  &&  0 == threadIdx.z )
	{
		expectedNode.setBaseType (NodeBaseType::FLUID) ;
	}

	if ( node.nodeType() != expectedNode.getBaseType() )
	{
		assert(0) ;
	}

	// Force test fail:
	//if ( threadIdx.x == 3  && threadIdx.y == 3 && threadIdx.z == 3) return ;

	int linearTileIndex = threadIdx.x + 4 * threadIdx.y + 16 * threadIdx.z ;

	for ( Direction::DirectionIndex q=0 ; q < 19 ; q++ )
	{
		node.f     (q) = linearTileIndex + q ;
		node.fPost (q) = -1 * (linearTileIndex + (int)q) ;
	}
}



template <TileDataArrangement DataArrangement>
void testTiledLatticeGPU()
{
	NodeLayout nodeLayout = createSolidNodeLayout( 4,4,4 ) ;
	nodeLayout.setNodeType(0, 0, 0, NodeBaseType::FLUID) ;
	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;
	TileLayout<StorageOnGPU> tileLayoutGPU( tileLayout ) ;

	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <D3Q19, double, StorageOnCPU   , DataArrangement> TLatticeCPU ;
	typedef TiledLattice <D3Q19, double, StorageOnGPU   , DataArrangement> TLatticeGPU ;
	typedef TiledLattice <D3Q19, double, StorageInKernel, DataArrangement> TLatticeKernel ;

	TLatticeCPU    tiledLatticeCPU (tileLayout, expandedNodeLayout, Settings()) ;
	TLatticeGPU    tiledLatticeGPU (tiledLatticeCPU, tileLayoutGPU) ;
	TLatticeKernel tiledLatticeKernel (tiledLatticeGPU) ;
	
	typedef typename TLatticeGPU::TileType TileType ;

	unsigned numberOfTiles = tiledLatticeCPU.getNOfTiles() ;
	unsigned tileWidth  = TileType::getNNodesPerEdge() ;
	unsigned tileHeight = TileType::getNNodesPerEdge() ;
	unsigned tileDepth  = TileType::getNNodesPerEdge() ;

	dim3 numBlocks  (numberOfTiles) ;
	dim3 numThreads (tileWidth, tileHeight, tileDepth) ;

	kernelTestTiledLatticeGPU<<< numBlocks, numThreads >>>
		(
			tiledLatticeKernel
		) ;
	ASSERT_EQ (cudaSuccess, cudaPeekAtLastError()) ;
	ASSERT_EQ (cudaSuccess, cudaDeviceSynchronize()) ;

	TLatticeCPU tiledLattice2 (tileLayout, expandedNodeLayout, Settings()) ;

	ASSERT_NO_THROW (tiledLatticeGPU.copyToCPU (tiledLattice2)) ;

	for (auto it = tiledLattice2.getBeginOfTiles() ; 
						it < tiledLattice2.getEndOfTiles() ;
						it++ )
	{
		auto tile = tiledLattice2.getTile( it ) ;

		constexpr unsigned edge = TileType::getNNodesPerEdge() ;

		std::cout << "f(E): " ;
		for (unsigned z=0 ; z < edge ; z++)
			for (unsigned y=0 ; y < edge ; y++)
				for (unsigned x=0 ; x < edge ; x++)
				{
					std::cout << tile.f(E)[z][y][x] << "\t" ;
				}
		std::cout << "\n" ;

		std::cout << "fPost(E): " ;
		for (unsigned z=0 ; z < edge ; z++)
			for (unsigned y=0 ; y < edge ; y++)
				for (unsigned x=0 ; x < edge ; x++)
				{
					std::cout << tile.fPost(E)[z][y][x] << "\t" ;
				}
		std::cout << "\n" ;

		for (unsigned z=0 ; z < edge ; z++)
			for (unsigned y=0 ; y < edge ; y++)
				for (unsigned x=0 ; x < edge ; x++)
				{
					auto node = tile.getNode(x,y,z) ;

					int linearIndex = x + 4 * y + 16 * z ;

					for ( Direction::DirectionIndex q=0 ; q < D3Q19::getQ() ; q++ )
					{
						EXPECT_EQ (node.f(q), linearIndex + q) <<
						"Difference for tile " << it << ", node x=" << x << ", y=" << y <<
						", z=" << z << ", q=" << q << "\n" ;
						EXPECT_EQ (node.fPost(q), -1 * (linearIndex + (int)q)) <<
						"Difference for tile " << it << ", node x=" << x << ", y=" << y <<
						", z=" << z << ", q=" << q << "\n" ;
					}
				}
	}
}


TEST( TiledLattice_XYZ, GPU )
{
	testTiledLatticeGPU <TileDataArrangement::XYZ> () ;
}



TEST( TiledLattice_OPT_1, GPU )
{
	testTiledLatticeGPU <TileDataArrangement::OPT_1> () ;
}



TEST (TiledLattice, modify)
{
	NodeLayout nodeLayout = createSolidNodeLayout (40,40,40) ;
	nodeLayout.setNodeType (0,0,0, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (4,4,4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (36,36,36, NodeBaseType::FLUID) ;
	
	TileLayout<StorageOnCPU> tileLayout (nodeLayout) ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	Settings settings("./test_data/cases/ruby_geometry_modifiers_shapes/") ;
	TLD3Q19 tiledLattice( tileLayout, expandedNodeLayout, settings ) ;


	ModificationRhoU modifications ;


	Coordinates coordinates(0,0,0) ;
	modifications.addUPhysical (coordinates, 1,2,3) ;
	modifications.addUBoundaryPhysical (Coordinates(1,1,1), 11,12,13) ;
	ASSERT_NO_THROW( tiledLattice.modify (modifications) ) ;
	
	auto tile0 = tiledLattice.getTile (0) ;
	
	EXPECT_EQ( tile0.u (Axis::X)[0][0][0], settings.transformVelocityPhysicalToLB(1.0) ) ;
	EXPECT_EQ( tile0.u (Axis::Y)[0][0][0], settings.transformVelocityPhysicalToLB(2.0) ) ;
	EXPECT_EQ( tile0.u (Axis::Z)[0][0][0], settings.transformVelocityPhysicalToLB(3.0) ) ;

	EXPECT_EQ( tile0.uBoundary (Axis::X)[1][1][1], settings.transformVelocityPhysicalToLB(11.0) ) ;
	EXPECT_EQ( tile0.uBoundary (Axis::Y)[1][1][1], settings.transformVelocityPhysicalToLB(12.0) ) ;
	EXPECT_EQ( tile0.uBoundary (Axis::Z)[1][1][1], settings.transformVelocityPhysicalToLB(13.0) ) ;

	modifications.addUPhysical (coordinates, 4,5,6) ;
	ASSERT_NO_THROW( tiledLattice.modify (modifications) ) ;
	EXPECT_EQ( tile0.u (Axis::X)[0][0][0], settings.transformVelocityPhysicalToLB(4.0) ) ;
	EXPECT_EQ( tile0.u (Axis::Y)[0][0][0], settings.transformVelocityPhysicalToLB(5.0) ) ;
	EXPECT_EQ( tile0.u (Axis::Z)[0][0][0], settings.transformVelocityPhysicalToLB(6.0) ) ;


	modifications.addRhoPhysical         (Coordinates(39,39,39), 20) ;
	modifications.addRhoBoundaryPhysical (Coordinates(39,39,39), 30) ;
	ASSERT_NO_THROW( tiledLattice.modify (modifications) ) ;

	auto tile2 = tiledLattice.getTile (2) ;

	EXPECT_EQ( tile2.rho        ()[3][3][3], settings.transformPressurePhysicalToVolumetricMassDensityLB(20) ) ;
	EXPECT_EQ( tile2.rhoBoundary()[3][3][3], settings.transformPressurePhysicalToVolumetricMassDensityLB(30) ) ;
}

