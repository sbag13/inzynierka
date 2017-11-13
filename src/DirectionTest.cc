#include "gtest/gtest.h"
#include "Direction.hpp"



using namespace microflow ;
using namespace std ;



#define DISABLE_COMPILER_WARNING(d) (void)d ;



TEST( Direction, straightNumber )
{
	unsigned counter = 0 ;
	
	for ( auto d : Direction::straight)
	{
		counter ++ ;
		DISABLE_COMPILER_WARNING(d) ;
	}

	EXPECT_EQ( 6u, counter ) ;
}



TEST( Direction, slantingNumber )
{
	unsigned counter = 0 ;
	
	for ( auto d : Direction::slanting)
	{
		counter ++ ;
		DISABLE_COMPILER_WARNING(d) ;
	}

	EXPECT_EQ( 12u, counter ) ;
}



TEST( Direction, D3Q19Number )
{
	unsigned counter = 0 ;
	
	for ( auto d : Direction::D3Q19)
	{
		counter ++ ;
		DISABLE_COMPILER_WARNING(d) ;
	}

	EXPECT_EQ( 18u, counter ) ;
}



TEST( Direction, D3Q27Number )
{
	unsigned counter = 0 ;
	
	for ( auto d : Direction::D3Q27)
	{
		counter ++ ;
		DISABLE_COMPILER_WARNING(d) ;
	}

	EXPECT_EQ( 26u, counter ) ;
}



TEST( Direction, get )
{
	EXPECT_EQ( Direction( Direction::EAST ).get(), Direction::EAST ) ;
	EXPECT_EQ( Direction( Direction::WEST ).get(), Direction::WEST ) ;
	EXPECT_EQ( Direction( Direction::NORTH ).get(), Direction::NORTH ) ;
	EXPECT_EQ( Direction( Direction::SOUTH ).get(), Direction::SOUTH ) ;
}



TEST( Direction, getX )
{
	EXPECT_EQ( Direction( Direction::EAST ).getX(),  1 ) ;
	EXPECT_EQ( Direction( Direction::WEST ).getX(), -1 ) ;

	EXPECT_EQ( Direction( Direction::NORTH ).getX(), 0 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH ).getX(), 0 ) ;

	EXPECT_EQ( Direction( Direction::TOP    ).getX(), 0 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM ).getX(), 0 ) ;


	EXPECT_EQ( Direction( Direction::EAST + Direction::NORTH ).getX(),  1 ) ;
	EXPECT_EQ( Direction( Direction::EAST + Direction::SOUTH ).getX(),  1 ) ;
	EXPECT_EQ( Direction( Direction::EAST + Direction::TOP   ).getX(),  1 ) ;
	EXPECT_EQ( Direction( Direction::EAST + Direction::BOTTOM).getX(),  1 ) ;

	EXPECT_EQ( Direction( Direction::EAST + Direction::NORTH  + Direction::TOP).getX(),  1 ) ;
	EXPECT_EQ( Direction( Direction::EAST + Direction::SOUTH  + Direction::TOP).getX(),  1 ) ;
	EXPECT_EQ( Direction( Direction::EAST + Direction::NORTH  + Direction::BOTTOM).getX(),  1 ) ;
	EXPECT_EQ( Direction( Direction::EAST + Direction::SOUTH  + Direction::BOTTOM).getX(),  1 ) ;

	EXPECT_EQ( Direction( Direction::WEST + Direction::NORTH ).getX(), -1 ) ;
	EXPECT_EQ( Direction( Direction::WEST + Direction::SOUTH ).getX(), -1 ) ;
	EXPECT_EQ( Direction( Direction::WEST + Direction::TOP   ).getX(), -1 ) ;
	EXPECT_EQ( Direction( Direction::WEST + Direction::BOTTOM).getX(), -1 ) ;

	EXPECT_EQ( Direction( Direction::WEST + Direction::NORTH + Direction::TOP).getX(), -1 ) ;
	EXPECT_EQ( Direction( Direction::WEST + Direction::SOUTH + Direction::TOP).getX(), -1 ) ;
	EXPECT_EQ( Direction( Direction::WEST + Direction::NORTH + Direction::BOTTOM).getX(), -1 ) ;
	EXPECT_EQ( Direction( Direction::WEST + Direction::SOUTH + Direction::BOTTOM).getX(), -1 ) ;
}



TEST( Direction, getY )
{
	EXPECT_EQ( Direction( Direction::EAST ).getY(), 0 ) ;
	EXPECT_EQ( Direction( Direction::WEST ).getY(), 0 ) ;

	EXPECT_EQ( Direction( Direction::NORTH ).getY(),  1 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH ).getY(), -1 ) ;

	EXPECT_EQ( Direction( Direction::TOP    ).getY(), 0 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM ).getY(), 0 ) ;

	EXPECT_EQ( Direction( Direction::NORTH + Direction::EAST  ).getY(),  1 ) ;
	EXPECT_EQ( Direction( Direction::NORTH + Direction::WEST  ).getY(),  1 ) ;
	EXPECT_EQ( Direction( Direction::NORTH + Direction::TOP   ).getY(),  1 ) ;
	EXPECT_EQ( Direction( Direction::NORTH + Direction::BOTTOM).getY(),  1 ) ;

	EXPECT_EQ( Direction( Direction::NORTH + Direction::EAST + Direction::TOP).getY(),  1 ) ;
	EXPECT_EQ( Direction( Direction::NORTH + Direction::WEST + Direction::TOP).getY(),  1 ) ;
	EXPECT_EQ( Direction( Direction::NORTH + Direction::EAST + Direction::BOTTOM).getY(),  1 ) ;
	EXPECT_EQ( Direction( Direction::NORTH + Direction::WEST + Direction::BOTTOM).getY(),  1 ) ;

	EXPECT_EQ( Direction( Direction::SOUTH + Direction::EAST  ).getY(), -1 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH + Direction::WEST  ).getY(), -1 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH + Direction::TOP   ).getY(), -1 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH + Direction::BOTTOM).getY(), -1 ) ;

	EXPECT_EQ( Direction( Direction::SOUTH + Direction::EAST + Direction::TOP).getY(),  -1 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH + Direction::WEST + Direction::TOP).getY(),  -1 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH + Direction::EAST + Direction::BOTTOM).getY(),  -1 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH + Direction::WEST + Direction::BOTTOM).getY(),  -1 ) ;
}



TEST( Direction, getZ )
{
	EXPECT_EQ( Direction( Direction::EAST ).getZ(), 0 ) ;
	EXPECT_EQ( Direction( Direction::WEST ).getZ(), 0 ) ;

	EXPECT_EQ( Direction( Direction::NORTH ).getZ(), 0 ) ;
	EXPECT_EQ( Direction( Direction::SOUTH ).getZ(), 0 ) ;

	EXPECT_EQ( Direction( Direction::TOP    ).getZ(),  1 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM ).getZ(), -1 ) ;

	EXPECT_EQ( Direction( Direction::TOP + Direction::EAST  ).getZ(),  1 ) ;
	EXPECT_EQ( Direction( Direction::TOP + Direction::WEST  ).getZ(),  1 ) ;
	EXPECT_EQ( Direction( Direction::TOP + Direction::NORTH ).getZ(),  1 ) ;
	EXPECT_EQ( Direction( Direction::TOP + Direction::SOUTH ).getZ(),  1 ) ;

	EXPECT_EQ( Direction( Direction::TOP + Direction::EAST + Direction::NORTH).getZ(),  1 ) ;
	EXPECT_EQ( Direction( Direction::TOP + Direction::WEST + Direction::NORTH).getZ(),  1 ) ;
	EXPECT_EQ( Direction( Direction::TOP + Direction::EAST + Direction::SOUTH).getZ(),  1 ) ;
	EXPECT_EQ( Direction( Direction::TOP + Direction::WEST + Direction::SOUTH).getZ(),  1 ) ;

	EXPECT_EQ( Direction( Direction::BOTTOM + Direction::EAST  ).getZ(), -1 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM + Direction::WEST  ).getZ(), -1 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM + Direction::NORTH ).getZ(), -1 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM + Direction::SOUTH ).getZ(), -1 ) ;

	EXPECT_EQ( Direction( Direction::BOTTOM + Direction::EAST + Direction::NORTH).getZ(), -1 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM + Direction::WEST + Direction::NORTH).getZ(), -1 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM + Direction::EAST + Direction::SOUTH).getZ(), -1 ) ;
	EXPECT_EQ( Direction( Direction::BOTTOM + Direction::WEST + Direction::SOUTH).getZ(), -1 ) ;
}



TEST( Direction, getIndexD3Q27 )
{
	for ( auto d : Direction::D3Q27)
	{
		unsigned directionIndex = Direction(d).getIndexD3Q27() ;

		EXPECT_EQ( d, Direction::D3Q27[ directionIndex ] ) ;
	}
}



TEST( Direction, setX_tooLarge )
{
	Direction d(Direction::EAST) ;

	ASSERT_DEATH( d.setX(2), "") ;
}



TEST( Direction, setX_tooSmall )
{
	Direction d(Direction::EAST) ;

	ASSERT_DEATH( d.setX(-2), "") ;
}



TEST( Direction, setX )
{
	{
		Direction d(Direction::EAST) ;

		EXPECT_EQ( d.getX(),  1 ) ;

		d.setX(0) ;
		EXPECT_EQ( d.getX(),  0 ) ;
		EXPECT_EQ( d.getY(),  0 ) ;
		EXPECT_EQ( d.getZ(),  0 ) ;

		d.setX(-1) ;
		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(),  0 ) ;
		EXPECT_EQ( d.getZ(),  0 ) ;

		d.setX(1) ;
		EXPECT_EQ( d.getX(),  1 ) ;
		EXPECT_EQ( d.getY(),  0 ) ;
		EXPECT_EQ( d.getZ(),  0 ) ;
	}
	{
		Direction d(Direction::EAST + Direction::SOUTH + Direction::BOTTOM ) ;

		EXPECT_EQ( d.getX(),  1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setX(0) ;
		EXPECT_EQ( d.getX(),  0 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setX(-1) ;
		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setX(1) ;
		EXPECT_EQ( d.getX(),  1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;
	}
}



TEST( Direction, setY )
{
	{
		Direction d(Direction::NORTH) ;

		EXPECT_EQ( d.getY(),  1 ) ;

		d.setY(0) ;
		EXPECT_EQ( d.getX(),  0 ) ;
		EXPECT_EQ( d.getY(),  0 ) ;
		EXPECT_EQ( d.getZ(),  0 ) ;

		d.setY(-1) ;
		EXPECT_EQ( d.getX(),  0 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(),  0 ) ;

		d.setY(1) ;
		EXPECT_EQ( d.getX(),  0 ) ;
		EXPECT_EQ( d.getY(),  1 ) ;
		EXPECT_EQ( d.getZ(),  0 ) ;
	}
	{
		Direction d(Direction::WEST + Direction::SOUTH + Direction::BOTTOM ) ;

		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setY(0) ;
		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(),  0 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setY(-1) ;
		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setY(1) ;
		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(),  1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;
	}
}



TEST( Direction, setZ )
{
	{
		Direction d(Direction::TOP) ;

		EXPECT_EQ( d.getZ(),  1 ) ;

		d.setZ(0) ;
		EXPECT_EQ( d.getX(),  0 ) ;
		EXPECT_EQ( d.getY(),  0 ) ;
		EXPECT_EQ( d.getZ(),  0 ) ;

		d.setZ(-1) ;
		EXPECT_EQ( d.getX(),  0 ) ;
		EXPECT_EQ( d.getY(),  0 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setZ(1) ;
		EXPECT_EQ( d.getX(),  0 ) ;
		EXPECT_EQ( d.getY(),  0 ) ;
		EXPECT_EQ( d.getZ(),  1 ) ;
	}
	{
		Direction d(Direction::WEST + Direction::SOUTH + Direction::BOTTOM ) ;

		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setZ(0) ;
		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(),  0 ) ;

		d.setZ(-1) ;
		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(), -1 ) ;

		d.setZ(1) ;
		EXPECT_EQ( d.getX(), -1 ) ;
		EXPECT_EQ( d.getY(), -1 ) ;
		EXPECT_EQ( d.getZ(),  1 ) ;
	}
}
