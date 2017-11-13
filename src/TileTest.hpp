#ifndef TILE_TEST_HPP
#define TILE_TEST_HPP



#include "TileDataArrangement.hpp"



namespace microflow
{



template <class Tile>
void fillTile (Tile & tile, unsigned & val)
{
	const unsigned edge = Tile::getNNodesPerEdge() ;

	typedef typename Tile::LatticeArrangementType LArrangement ;

	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				auto node = tile.getNode (x,y,z) ;

				for (unsigned i=0 ; i < LArrangement::getQ() ; i++)
				{
					auto direction = LArrangement::c[i] ;
					
					node.f( direction ) = ++val ; 
					node.fPost( direction ) = ++val ; 
				}

				node.rho() = ++val ; 
				node.rho0() = ++val ; 
				node.rhoBoundary() = ++val ; 

				node.u( Axis::X ) = ++val ; 
				node.u( Axis::Y ) = ++val ; 
				node.u( Axis::Z ) = ++val ; 
				node.uT0( Axis::X ) = ++val ; 
				node.uT0( Axis::Y ) = ++val ; 
				node.uT0( Axis::Z ) = ++val ; 
				node.uBoundary( Axis::X ) = ++val ; 
				node.uBoundary( Axis::Y ) = ++val ; 
				node.uBoundary( Axis::Z ) = ++val ; 
			}
}



template <class Tile>
void checkTile (Tile & tile, unsigned & val)
{
	const unsigned edge = Tile::getNNodesPerEdge() ;

	typedef typename Tile::LatticeArrangementType LArrangement ;

	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				for (unsigned i=0 ; i < LArrangement::getQ() ; i++)
				{
					// For tiles arranged in different way than XYZ the values
					// of some f_i are stored in different order.
					if (TileDataArrangement::XYZ == Tile::tileDataArrangement)
					{
						auto direction = LArrangement::c[i] ;

						ASSERT_EQ( tile.f( direction )[z][y][x], ++val ) \
							<< "x=" << x <<", y=" << y << ", z=" << z << ", d=" << i \
							<< ", val=" << val-1 << "\n" ; 
						ASSERT_EQ( tile.fPost( direction )[z][y][x], ++val ) \
							<< "x=" << x <<", y=" << y << ", z=" << z << ", d=" << i \
							<< ", val=" << val-1 << "\n" ; 
					}
					else
					{
						val += 2 ;
					}
				}

				ASSERT_EQ( tile.rho()[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.rho0()[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.rhoBoundary()[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

				ASSERT_EQ( tile.u( Axis::X )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.u( Axis::Y )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.u( Axis::Z )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

				ASSERT_EQ( tile.uT0( Axis::X )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.uT0( Axis::Y )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.uT0( Axis::Z )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

				ASSERT_EQ( tile.uBoundary( Axis::X )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.uBoundary( Axis::Y )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( tile.uBoundary( Axis::Z )[z][y][x], ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
			}
}



template <class Tile>
void checkTileNodes (Tile & tile, unsigned & val)
{
	const unsigned edge = Tile::getNNodesPerEdge() ;

	typedef typename Tile::LatticeArrangementType LArrangement ;

	for (unsigned z=0 ; z < edge ; z++)
		for (unsigned y=0 ; y < edge ; y++)
			for (unsigned x=0 ; x < edge ; x++)
			{
				auto node = tile.getNode(x,y,z) ;

				for (unsigned i=0 ; i < LArrangement::getQ() ; i++)
				{
					Direction::D direction = LArrangement::c[i] ;
					Direction::DirectionIndex di = LArrangement::getIndex (direction) ;
					//Direction::DirectionIndex di = i ; // Both versions work.
					
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

				ASSERT_EQ( node.uT0( X ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.uT0( Y ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.uT0( Z ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 

				ASSERT_EQ( node.uBoundary( Axis::X ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.uBoundary( Axis::Y ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
				ASSERT_EQ( node.uBoundary( Axis::Z ), ++val ) \
					<< "x=" << x <<", y=" << y << ", z=" << z \
					<< ", val=" << val-1 << "\n" ; 
			}
}



}



#endif
