#include "gtest/gtest.h"
#include "LatticeCalculator.hpp"
#include "FluidModels.hpp"
#include "CollisionModels.hpp"
#include "NodeLayoutTest.hpp"
#include "TiledLatticeTest.hpp"
#include <sstream>



using namespace microflow ;
using namespace std ;



/*
 WARNING. In the new version of GPU propagation there is a different data flow than
					in RS code. GPU propagation always either reads only F_POST and writes 
					only F or reads only F and writes only F_POST. After one LBM step F and
					F_POST are exchanged.
					In RS code data is read and written to f_post, f is only temporary.

					This organization requires, that for propagation tests F and F_POST
					values have to be equal, because for nodes, which are not propagated
					from neighbors we need to read old F value from F_POST (or vice versa).
					In normal code this requirement is met by F and F_POST exchange after
					each LBM iteration (look at tests ???).
 */
template <class TiledLattice>
static inline
void fillGeometryWithConsecutiveValuesEqFPostF(TiledLattice & tiledLattice)
{
	unsigned val = 0 ;

	// Fill f_i functions with different values
	for (auto t = tiledLattice.getBeginOfTiles() ; t < tiledLattice.getEndOfTiles() ; t++)
	{
		auto tile = tiledLattice.getTile( t ) ;
		
		constexpr unsigned Edge = tile.getNNodesPerEdge() ;
		for (unsigned z=0 ; z < Edge ; z++)
			for (unsigned y=0 ; y < Edge ; y++)
				for (unsigned x=0 ; x < Edge ; x++)
				{
					typedef typename TiledLattice::LatticeArrangementType LArrangement ;

					for (unsigned q=0 ; q < LArrangement::getQ() ; q++)
					{
						val++ ;
						tile.f()[q][z][y][x]     = val ;
						tile.fPost()[q][z][y][x] = val ;
					}
				}
	}
}



template <class Node>
static inline
void printCompared (Node & n1, Node & n2)
{
	typedef typename Node::LatticeArrangementType LArrangement ;

	std::cout << "n1 at x=" << n1.getNodeInTileX() 
						<< ",y=" << n1.getNodeInTileY() << ",z=" << n1.getNodeInTileZ()
						<< "\n" ;
	std::cout << "n2 at x=" << n2.getNodeInTileX() 
						<< ",y=" << n2.getNodeInTileY() << ",z=" << n2.getNodeInTileZ()
						<< "\n" ;

	std::cout << "n1: " << n1.nodeType() << "\n" ;
	std::cout << "n2: " << n2.nodeType() << "\n" ;
	
	std::cout << "n1 f: " ;
	for (Direction::DirectionIndex q=0 ; q < LArrangement::getQ() ; q++)
	{
		std::cout << n1.f(q) << ", " ;
	}
	std::cout << "\n" ;

	std::cout << "n2 f: " ;
	for (Direction::DirectionIndex q=0 ; q < LArrangement::getQ() ; q++)
	{
		std::cout << n2.f(q) << ", " ;
	}
	std::cout << "\n" ;

	std::cout << "n1 fPost: " ;
	for (Direction::DirectionIndex q=0 ; q < LArrangement::getQ() ; q++)
	{
		std::cout << n1.fPost(q) << ", " ;
	}
	std::cout << "\n" ;

	std::cout << "n2 fPost: " ;
	for (Direction::DirectionIndex q=0 ; q < LArrangement::getQ() ; q++)
	{
		std::cout << n2.fPost(q) << ", " ;
	}
	std::cout << "\n" ;
}



template <class TiledLattice>
static inline
bool compare (const TiledLattice & tl1, const TiledLattice & tl2)
{
	bool result = true ;

	if (tl1.getNOfTiles() != tl2.getNOfTiles())
	{
		std::cout << "Number of tiles differ: tl1 has " << tl1.getNOfTiles()
							<< " tiles, tl2 has " << tl2.getNOfTiles() << " tiles.\n" ;
		return false ;
	}

#define LOCATION "tile=" << ti << ", x=" << x << ", y=" << y << ", z=" << z
#define LOC_Q LOCATION << ", q=" << q
								 

	typedef typename TiledLattice::LatticeArrangementType LArrangement ;

	for (auto ti = tl1.getBeginOfTiles() ; ti < tl1.getEndOfTiles() ; ti++)
	{
		auto t1 = tl1.getTile (ti) ;
		auto t2 = tl2.getTile (ti) ;

		constexpr unsigned edge = t1.getNNodesPerEdge() ;

		for (unsigned z=0 ; z < edge ; z++)
			for (unsigned y=0 ; y < edge ; y++)
				for (unsigned x=0 ; x < edge ; x++)
				{
					bool nodesAreEqual = true ;

					auto n1 = t1.getNode (x,y,z) ;
					auto n2 = t2.getNode (x,y,z) ;

					EXPECT_EQ (n1.nodeType(), n2.nodeType()) ;

					if (n1.nodeType() != n2.nodeType())
					{
						nodesAreEqual = false ;
					}

					// WARNING. Due to optimizations values in SOLID nodes are not updated.
					// 					Thus, simple comparison of raw data results in differences, 
					//					because for SOLID nodes some values may be uninitialized.
					if (not n1.nodeType().isSolid())
					{
						for (Direction::DirectionIndex q=0 ; q < LArrangement::getQ() ; q++)
						{
							if (
									n1.f (q) != n2.f (q) ||
									n1.fPost (q) != n2.fPost (q)
								 )
							{
								nodesAreEqual = false ;
							}
						}

						if (not nodesAreEqual)
						{
							std::cout << "Tile " << ti <<"\n" ;
							printCompared (n1,n2) ;

							result = false ;
						}
					}
				}
	}

	return result ;
}


template <TileDataArrangement DataArrangement>
static inline
void testPropagateGPU( TileLayout<StorageOnCPU> & tileLayout, 
											 ExpandedNodeLayout & expandedNodeLayout )
{
	typedef D3Q19 LArrangement ;
	typedef double DType ;

	typedef TiledLattice <LArrangement, DType, StorageOnCPU, DataArrangement> TLatticeCPU ;
	typedef TiledLattice <LArrangement, DType, StorageOnGPU, DataArrangement> TLatticeGPU ;

	TileLayout<StorageOnGPU> tileLayoutGPU( tileLayout ) ;
	
	TLatticeCPU tiledLatticeCPU (tileLayout, expandedNodeLayout, Settings()) ;

	TLatticeCPU tiledLatticeCPU2 (tileLayout, expandedNodeLayout, Settings()) ;
	TLatticeGPU tiledLatticeGPU  (tiledLatticeCPU2, tileLayoutGPU) ;

	typedef LatticeCalculator <FluidModelIncompressible, CollisionModelBGK,
														 LArrangement, DType, StorageOnCPU, DataArrangement> LC ;	
	typedef LatticeCalculator <FluidModelIncompressible, CollisionModelBGK,
														 LArrangement, DType, StorageOnGPU, DataArrangement> LC_GPU ;	

	DType u0LB[ LArrangement::getD() ] ;
	u0LB[0] = 1.0 ;
	u0LB[1] = 1.0 ;
	u0LB[2] = 1.0 ;

	LC     latticeCalculator (1.0, u0LB, 1.0,
														NodeType (NodeBaseType::VELOCITY_0,
															 	PlacementModifier::EXTERNAL_EDGE_PRESSURE_TANGENTIAL),
														NodeType (NodeBaseType::VELOCITY_0,
															 	PlacementModifier::EXTERNAL_CORNER_PRESSURE_TANGENTIAL)
													 ) ;
	LC_GPU latticeCalculatorGPU (1.0, u0LB, 1.0,
														 	 NodeType (NodeBaseType::VELOCITY_0,
														 	 		PlacementModifier::EXTERNAL_EDGE_PRESSURE_TANGENTIAL),
														 	 NodeType (NodeBaseType::VELOCITY_0,
														 	 		PlacementModifier::EXTERNAL_CORNER_PRESSURE_TANGENTIAL)
														  ) ;

	fillGeometryWithConsecutiveValuesEqFPostF( tiledLatticeCPU  ) ;

	tiledLatticeGPU.copyFromCPU( tiledLatticeCPU ) ;
	tiledLatticeGPU.copyToCPU() ;
	EXPECT_TRUE (compare (tiledLatticeCPU, tiledLatticeCPU2)) ;

	{
		std::cout << "Before propagation: tile 0:\n" ;
		auto t1 = tiledLatticeCPU.getTile(0) ;
		auto t2 = tiledLatticeCPU2.getTile(0) ;
		auto n1 = t1.getNode (0,0,0) ;
		auto n2 = t2.getNode (0,0,0) ;
		printCompared (n1,n2) ;
	}

	// WARNING: Explicit setting needed, since we force non-standard computations order.
	tiledLatticeGPU.setValidCopyIDToFPost() ; 

	EXPECT_NO_THROW( latticeCalculator.propagate( tiledLatticeCPU ) ) ;
	EXPECT_NO_THROW( latticeCalculatorGPU.propagate( tiledLatticeGPU ) ) ;
	tiledLatticeGPU.copyToCPU() ;
	EXPECT_TRUE (compare (tiledLatticeCPU, tiledLatticeCPU2)) ;

	{
		std::cout << "After propagation: tile 0:\n" ;
		auto t1 = tiledLatticeCPU.getTile(0) ;
		auto t2 = tiledLatticeCPU2.getTile(0) ;
		auto n1 = t1.getNode (0,0,0) ;
		auto n2 = t2.getNode (0,0,0) ;
		printCompared (n1,n2) ;
	}
}



template <class NodeLayoutModifier, TileDataArrangement DataArrangement>
static inline
void testPropagateFullSceneGPU( Size sceneSizeInTiles )
{
	ASSERT_NO_THROW
	(
		NodeLayout nodeLayout = createFluidNodeLayout( 
																		sceneSizeInTiles.getX()*4, 
																		sceneSizeInTiles.getY()*4, 
																		sceneSizeInTiles.getZ()*4
																		) ;

		NodeLayoutModifier::modify( nodeLayout ) ;

		TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;


		testPropagateGPU <DataArrangement> (tileLayout, expandedNodeLayout) ;
	) ;
}



class NoModify
{
	public:
		static void modify( NodeLayout & nodeLayout ) {}
} ;



TEST (LatticeCalculator_XYZ, propagateSingleTileFluidGPU_D3Q19)
{
	testPropagateFullSceneGPU <NoModify, TileDataArrangement::XYZ> (Size(1,1,1)) ;
}



TEST (LatticeCalculator_OPT_1, propagateSingleTileFluidGPU_D3Q19)
{
	testPropagateFullSceneGPU <NoModify, TileDataArrangement::OPT_1> (Size(1,1,1)) ;
}



class AllTypes
{
	public:
		static void modify( NodeLayout & nodeLayout )
		{
			Size s = nodeLayout.getSize() ;
			
			unsigned nodeTypeIndex = 0 ;

			for (unsigned z=0 ; z < s.getDepth() ; z++ )
				for (unsigned y=0 ; y < s.getHeight() ; y++)
					for (unsigned x=0 ; x < s.getWidth() ; x++)
					{
						nodeLayout.setNodeType( x,y,z, static_cast<NodeBaseType>(nodeTypeIndex) ) ;

						nodeTypeIndex = (nodeTypeIndex + 1) % static_cast<unsigned>(NodeBaseType::MARKER) ;
					}
		}
} ;



TEST (LatticeCalculator_XYZ, propagateSingleTileAllTypesGPU_D3Q19)
{
	testPropagateFullSceneGPU <AllTypes, TileDataArrangement::XYZ>( Size(1,1,1) ) ;
}



TEST (LatticeCalculator_OPT_1, propagateSingleTileAllTypesGPU_D3Q19)
{
	testPropagateFullSceneGPU <AllTypes, TileDataArrangement::OPT_1>( Size(1,1,1) ) ;
}



static inline
void fillTileWithFluidNodes( NodeLayout & nodeLayout, 
														 Coordinates tileCoordinates)
{
	for (unsigned z=0 ; z < 4 ; z++)
		for (unsigned y=0 ; y < 4 ; y++)
			for (unsigned x=0 ; x < 4 ; x++)
			{
				nodeLayout.setNodeType( tileCoordinates + Coordinates(x,y,z), 
																NodeBaseType::FLUID ) ;
			}
}



template <TileDataArrangement DataArrangement>
static inline
void testPropagateTwoTilesGPU( Coordinates baseTile, Coordinates neighborTile )
{
	NodeLayout nodeLayout = createSolidNodeLayout( 3*4, 3*4, 3*4 ) ;

	fillTileWithFluidNodes( nodeLayout, baseTile ) ;
	fillTileWithFluidNodes( nodeLayout, neighborTile ) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;
	EXPECT_EQ( 128u, tilingStatistic.getNFluidNodes() ) ;
	EXPECT_EQ( 2u, tilingStatistic.getNNonEmptyTiles() ) ;

	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;


	testPropagateGPU <DataArrangement> (tileLayout, expandedNodeLayout) ;
}



TEST (LatticeCalculator_XYZ, propagateTwoTilesFluidGPU_D3Q19_TOP)
{
	testPropagateTwoTilesGPU <TileDataArrangement::XYZ> (Coordinates(4,4,4), Coordinates(4,4,8)) ;
}



TEST (LatticeCalculator_OPT_1, propagateTwoTilesFluidGPU_D3Q19_TOP)
{
	testPropagateTwoTilesGPU <TileDataArrangement::OPT_1> (Coordinates(4,4,4), Coordinates(4,4,8)) ;
}



TEST (LatticeCalculator_XYZ, propagateTwoTilesFluidGPU_D3Q19_BOTTOM)
{
	testPropagateTwoTilesGPU <TileDataArrangement::XYZ> (Coordinates(4,4,4), Coordinates(4,4,0)) ;
}



TEST (LatticeCalculator_OPT_1, propagateTwoTilesFluidGPU_D3Q19_BOTTOM)
{
	testPropagateTwoTilesGPU <TileDataArrangement::OPT_1> (Coordinates(4,4,4), Coordinates(4,4,0)) ;
}



TEST (LatticeCalculator_XYZ, propagateTwoTilesFluidGPU_D3Q19_EAST)
{
	testPropagateTwoTilesGPU <TileDataArrangement::XYZ> (Coordinates(4,4,4), Coordinates(8,4,4)) ;
}



TEST (LatticeCalculator_OPT_1, propagateTwoTilesFluidGPU_D3Q19_EAST)
{
	testPropagateTwoTilesGPU <TileDataArrangement::OPT_1> (Coordinates(4,4,4), Coordinates(8,4,4)) ;
}



TEST (LatticeCalculator_XYZ, propagateTwoTilesFluidGPU_D3Q19_WEST)
{
	testPropagateTwoTilesGPU <TileDataArrangement::XYZ> (Coordinates(4,4,4), Coordinates(0,4,4)) ;
}



TEST (LatticeCalculator_OPT_1, propagateTwoTilesFluidGPU_D3Q19_WEST)
{
	testPropagateTwoTilesGPU <TileDataArrangement::OPT_1> (Coordinates(4,4,4), Coordinates(0,4,4)) ;
}



TEST (LatticeCalculator_XYZ, propagateTwoTilesFluidGPU_D3Q19_NORTH)
{
	testPropagateTwoTilesGPU <TileDataArrangement::XYZ> (Coordinates(4,4,4), Coordinates(4,8,4)) ;
}



TEST (LatticeCalculator_OPT_1, propagateTwoTilesFluidGPU_D3Q19_NORTH)
{
	testPropagateTwoTilesGPU <TileDataArrangement::OPT_1> (Coordinates(4,4,4), Coordinates(4,8,4)) ;
}



TEST (LatticeCalculator_XYZ, propagateTwoTilesFluidGPU_D3Q19_SOUTH)
{
	testPropagateTwoTilesGPU <TileDataArrangement::XYZ> (Coordinates(4,4,4), Coordinates(4,0,4)) ;
}



TEST (LatticeCalculator_OPT_1, propagateTwoTilesFluidGPU_D3Q19_SOUTH)
{
	testPropagateTwoTilesGPU <TileDataArrangement::OPT_1> (Coordinates(4,4,4), Coordinates(4,0,4)) ;
}



template <TileDataArrangement DataArrangement>
static inline
void testPropagateTilesFluidGPU_D3Q19_NE()
{
	NodeLayout nodeLayout = createSolidNodeLayout( 3*4, 3*4, 3*4 ) ;

	fillTileWithFluidNodes( nodeLayout, Coordinates(4,4,4) ) ;

	fillTileWithFluidNodes( nodeLayout, Coordinates(8,4,4) ) ;
	fillTileWithFluidNodes( nodeLayout, Coordinates(4,8,4) ) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;
	EXPECT_EQ( 3u * 64u, tilingStatistic.getNFluidNodes() ) ;
	EXPECT_EQ( 3u, tilingStatistic.getNNonEmptyTiles() ) ;

	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;


	testPropagateGPU <DataArrangement> (tileLayout, expandedNodeLayout) ;
}



TEST (LatticeCalculator_XYZ, propagateTilesFluidGPU_D3Q19_NE)
{
	testPropagateTilesFluidGPU_D3Q19_NE <TileDataArrangement::XYZ> () ;
}



TEST (LatticeCalculator_OPT_1, propagateTilesFluidGPU_D3Q19_NE)
{
	testPropagateTilesFluidGPU_D3Q19_NE <TileDataArrangement::OPT_1> () ;
}



template <TileDataArrangement DataArrangement>
static inline
void testPropagateTilesFluidGPU_D3Q19_allStraight()
{
	NodeLayout nodeLayout = createSolidNodeLayout( 3*4, 3*4, 3*4 ) ;

	fillTileWithFluidNodes( nodeLayout, Coordinates(4,4,4) ) ;

	fillTileWithFluidNodes( nodeLayout, Coordinates(0,4,4) ) ;
	fillTileWithFluidNodes( nodeLayout, Coordinates(8,4,4) ) ;

	fillTileWithFluidNodes( nodeLayout, Coordinates(4,0,4) ) ;
	fillTileWithFluidNodes( nodeLayout, Coordinates(4,8,4) ) ;

	fillTileWithFluidNodes( nodeLayout, Coordinates(4,4,0) ) ;
	fillTileWithFluidNodes( nodeLayout, Coordinates(4,4,8) ) ;

	TileLayout<StorageOnCPU> tileLayout( nodeLayout ) ;

	TilingStatistic tilingStatistic = tileLayout.computeTilingStatistic() ;
	EXPECT_EQ( 7u * 64u, tilingStatistic.getNFluidNodes() ) ;
	EXPECT_EQ( 7u, tilingStatistic.getNNonEmptyTiles() ) ;

	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;


	testPropagateGPU <DataArrangement> (tileLayout, expandedNodeLayout) ;
}



TEST (LatticeCalculator_XYZ, propagateTilesFluidGPU_D3Q19_allStraight)
{
	testPropagateTilesFluidGPU_D3Q19_allStraight <TileDataArrangement::XYZ> () ;
}



TEST (LatticeCalculator_OPT_1, propagateTilesFluidGPU_D3Q19_allStraight)
{
	testPropagateTilesFluidGPU_D3Q19_allStraight <TileDataArrangement::OPT_1> () ;
}



TEST (LatticeCalculator_XYZ, propagateFluidGPU_D3Q19_11x11x11_tiles)
{
	testPropagateFullSceneGPU <NoModify, TileDataArrangement::XYZ> (Size(11,11,11)) ;
}



TEST (LatticeCalculator_OPT_1, propagateFluidGPU_D3Q19_11x11x11_tiles)
{
	testPropagateFullSceneGPU <NoModify, TileDataArrangement::OPT_1> (Size(11,11,11)) ;
}



TEST (LatticeCalculator_XYZ, propagateAllTypesGPU_D3Q19_11x11x11_tiles)
{
	testPropagateFullSceneGPU <AllTypes, TileDataArrangement::XYZ> (Size(11,11,11)) ;
}



TEST (LatticeCalculator_OPT_1, propagateAllTypesGPU_D3Q19_11x11x11_tiles)
{
	testPropagateFullSceneGPU <AllTypes, TileDataArrangement::OPT_1> (Size(11,11,11)) ;
}



template <TileDataArrangement DataArrangement>
void testComputeRhoForBB2Nodes()
{
	auto nodeLayout = createFluidNodeLayout (4*3,4*3,4*3) ;
	
	TileLayout <StorageOnCPU> tileLayout (nodeLayout) ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;
	
	typedef TiledLattice <D3Q19, double, StorageOnCPU, DataArrangement> 
		TLattice ;
		
	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;

	double val = 1.0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TLattice::TileType::DefaultNodeType & node,
				 Coordinates & globCoord)
		{
			node.rho() = val ;
			val += 1.0 ;

			if (globCoord == Coordinates (1,1,1))
			{
				node.nodeType().setBaseType (NodeBaseType::BOUNCE_BACK_2) ;
			}
		}
	) ;

	typedef LatticeCalculator <FluidModelIncompressible, CollisionModelBGK, 
														 D3Q19, double, StorageOnCPU, 
														 DataArrangement>
						LC ;

	double u0LB[3] ;
	LC latticeCalculator (1.0, u0LB, 1.0, 
												NodeType (NodeBaseType::VELOCITY_0,
														PlacementModifier::EXTERNAL_EDGE_PRESSURE_TANGENTIAL),
												NodeType (NodeBaseType::VELOCITY_0,
														PlacementModifier::EXTERNAL_CORNER_PRESSURE_TANGENTIAL)
											 ) ;
	ASSERT_NO_THROW (latticeCalculator.computeRhoForBB2Nodes (tiledLattice)) ;

	val = 1.0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TLattice::TileType::DefaultNodeType & node,
				 Coordinates & globCoord)
		{
			double expectedVal = val ;
			val += 1.0 ;

			if (globCoord == Coordinates (1,1,1))
			{
				expectedVal = 
					(2 + 5+6+7 + 10 + 
					 17+18+19 + 21+23 + 25+26+27 + 
					 34 + 37+38+39 + 42) / 18.0 ;
			}

			EXPECT_EQ (expectedVal, node.rho()) << "Difference for " << globCoord << "\n" ;
		}
	) ;

	
	auto tile0 = tiledLattice.getTile (0) ;
	tile0.getNode (1,1,0).nodeType().setBaseType (NodeBaseType::BOUNCE_BACK_2) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;

	ASSERT_NO_THROW (latticeCalculator.computeRhoForBB2Nodes (tiledLattice)) ;

	EXPECT_EQ (tile0.getNode (1,1,0).rho(), 
			(1+2+3 + 5+7 + 9+10+11 + 18 + 21+23 + 26) / 12.0) ;

	EXPECT_EQ (tile0.getNode (1,1,1).rho(), 
			(2 + 5+7 + 10 + 
			 17+18+19 + 21+23 + 25+26+27 + 
			 34 + 37+38+39 + 42) / 17.0) ;

	
	// Nodes at tile edges
	tile0.getNode (1,1,0).nodeType().setBaseType (NodeBaseType::SOLID) ;
	tile0.getNode (1,1,1).nodeType().setBaseType (NodeBaseType::SOLID) ;

	auto tile1 = tile0.getNeighbor (E) ;
	tile1.getNode (0,0,0).nodeType().setBaseType (NodeBaseType::BOUNCE_BACK_2) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;

	ASSERT_NO_THROW (latticeCalculator.computeRhoForBB2Nodes (tiledLattice)) ;

	auto tile1Index = tile1.getCurrentTileIndex() ;
	logger << "tile1 index = " << tile1Index << "\n" ;
	EXPECT_EQ (tile1.getNode (0,0,0).rho(), 
			(4 + 8 + 20 + (2 + 5+6 + 17+18 + 21 + 6*tile1Index*64) ) / 9.0) ;
}



TEST (LatticeCalculator_XYZ, computeRhoForBB2Nodes)
{
	testComputeRhoForBB2Nodes <TileDataArrangement::XYZ>() ;
}



TEST (LatticeCalculator_OPT_1, computeRhoForBB2Nodes)
{
	testComputeRhoForBB2Nodes <TileDataArrangement::OPT_1>() ;
}



