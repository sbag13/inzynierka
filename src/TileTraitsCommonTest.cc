#include "gtest/gtest.h"

#include "TileTraitsCommon.hpp"



using namespace microflow ;



class TTCtst : public TileTraitsCommon<4,2>
{} ;



TEST( TileTraitsCommon, getNNodesPerTile )
{
	EXPECT_EQ( ( TileTraitsCommon<4,2>::getNNodesPerEdge() ), 4u ) ;
	EXPECT_EQ( ( TTCtst::getNNodesPerEdge() ), 4u ) ;
}
