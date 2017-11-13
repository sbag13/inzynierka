#ifndef NODE_FROM_TILE_TEST_HPP
#define NODE_FROM_TILE_TEST_HPP



#include "NodeFromTile.hpp"
#include "Logger.hpp"
#include "BaseIO.hpp"
#include "Direction.hpp"



namespace microflow
{



/*
	FIXME: Unfinished !!!
 */
template <class Tile, DataStorageMethod DataStorage>
inline
bool areEqual (NodeFromTile <Tile,DataStorage> & node1, 
							 NodeFromTile <Tile,DataStorage> & node2)
{

#define COMPARE(attribute)                                    \
if (node1.attribute() != node2.attribute())                   \
{                                                             \
	logger << #attribute " differ: "                            \
				 << node1.attribute() << ", " << node2.attribute()    \
				 << "\n" ;                                            \
	return false ;                                              \
}

	COMPARE (nodeType) ;
	COMPARE (rho) ;
	COMPARE (rho0) ;
	COMPARE (rhoBoundary) ;

#undef COMPARE

	for (Direction::DirectionIndex q=0 ;
			 q < Tile::LatticeArrangementType::getQ() ;
			 q++)
	{
		if (node1.f (q) != node2.f (q))
		{
			logger << buildFArrayName<typename Tile::LatticeArrangementType> ("f", q)
						 << " differ: " << node1.f (q) << ", " << node2.f (q)
						 << "\n" ;
			return false ;
		}
	}

	return true ;
}



}



#endif
