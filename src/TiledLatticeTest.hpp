#ifndef TILED_LATTICE_TEST_HPP
#define TILED_LATTICE_TEST_HPP



#include "TileLayoutTest.hpp"
#include "TiledLattice.hpp"
#include "Logger.hpp"



namespace microflow
{



typedef TiledLattice <D3Q19, double, StorageOnCPU, TileDataArrangement::XYZ>	TLD3Q19 ;
typedef TiledLattice <D3Q19, double, StorageOnGPU, TileDataArrangement::XYZ>	TLD3Q19GPU ;

typedef Tile <D3Q19, double, DEFAULT_3D_TILE_EDGE, StorageOnCPU, TileDataArrangement::XYZ>   
					TileD3Q19 ;



/*
	FIXME: In file LatticeCalculatorTest.cc file there are 
				 almost identical functions. 
*/
template <class TiledLattice>
inline
void fillWithConsecutiveValues (TiledLattice & tiledLattice)
{
	typename TiledLattice::DataTypeType val = 1.0 ;

	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				for (Direction::DirectionIndex q=0 ; 
						 q < TiledLattice::LatticeArrangementType::getQ() ;
						 q++)
				{
					node.f     (q) = val ; val += 1.0 ;
					node.fPost (q) = val ; val += 1.0 ;
				}

				for (unsigned c=0 ;
						 c < TiledLattice::LatticeArrangementType::getD() ;
						 c++)
				{
					node.u         (c) = val ; val += 1.0 ;
					node.uBoundary (c) = val ; val += 1.0 ;
					node.uT0       (c) = val ; val += 1.0 ;
				}

				node.rho        () = val ; val += 1.0 ;
				node.rho0       () = val ; val += 1.0 ;
				node.rhoBoundary() = val ; val += 1.0 ;
			}
	) ;
}



/*
	FIXME: Unfinished !!!
 */
template <class TiledLattice>
inline
bool areEqual (TiledLattice const & tLattice1, TiledLattice const & tLattice2)
{
	auto const t1NOfTiles = tLattice1.getNOfTiles() ;
	auto const t2NOfTiles = tLattice2.getNOfTiles() ;

	if (t1NOfTiles != t2NOfTiles)
	{
		logger << "Number of tiles differ: " << t1NOfTiles << " != " 
					 << t2NOfTiles << "\n" ;
		return false ;
	}

	auto t1Iterator = tLattice1.getBeginOfTiles() ;
	auto t2Iterator = tLattice2.getBeginOfTiles() ;

	for ( ; t1Iterator < tLattice1.getEndOfTiles() && 
					t2Iterator < tLattice2.getEndOfTiles() 
				; t1Iterator++, t2Iterator++)
	{
		auto t1 = tLattice1.getTile (t1Iterator) ;
		auto t2 = tLattice2.getTile (t2Iterator) ;

		auto t1Corner = t1.getCornerPosition() ;
		auto t2Corner = t2.getCornerPosition() ;

		if (t1Corner != t2Corner)
		{
			logger << "Position for tiles " 
						 << t1Iterator << ", " << t2Iterator
						 << " differ: " << t1Corner << ", " << t2Corner
						 << "\n" ;
		}

		auto constexpr edge = TiledLattice::getNNodesPerTileEdge() ;

		for (unsigned z=0 ; z < edge ; z++)
			for (unsigned y=0 ; y < edge ; y++)
				for (unsigned x=0 ; x < edge ; x++)
				{
					auto n1 = t1.getNode (x,y,z) ;
					auto n2 = t2.getNode (x,y,z) ;

					if (not areEqual (n1,n2))
					{
						logger << "Nodes differ in tile " << t1Iterator
									 << " at (x=" << x << ",y=" << y << ",z=" << z << ")"
									 << ", global coordinates=" << t1Corner + Coordinates (x,y,z)
									 << "\n" ;

						return false ;
					}
				}
	}
	
	return true ;
}



}



#endif
