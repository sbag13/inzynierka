#ifndef DIRECTION_HPP
#define DIRECTION_HPP



#include <ostream>

#include "PackedDirectionVector.hpp"
#include "cudaPrefix.hpp"



namespace microflow
{



#define DIRECTION_TYPE char



class Direction : private PackedDirectionVector<DIRECTION_TYPE>
{
	public:

		typedef PackedDirectionVector<DIRECTION_TYPE>::InternalStorageType   D ;

		static constexpr D SELF   = 0 ;

		static constexpr D EAST   = 0b00000001 ; // x = +1
		static constexpr D WEST   = 0b00000011 ; // x = -1

		static constexpr D NORTH  = 0b00000100 ; // y = +1
		static constexpr D SOUTH  = 0b00001100 ; // y = -1

		static constexpr D TOP    = 0b00010000 ; // z = +1
		static constexpr D BOTTOM = 0b00110000 ; // z = -1

#define STRAIGHT EAST, NORTH, WEST, SOUTH, TOP, BOTTOM

#define SLANTING 	                                              \
				EAST + NORTH, EAST + SOUTH, EAST + TOP, EAST + BOTTOM,  \
				WEST + NORTH, WEST + SOUTH, WEST + TOP, WEST + BOTTOM,  \
				NORTH + TOP, NORTH + BOTTOM,                            \
				SOUTH + TOP, SOUTH + BOTTOM
				
#define CORNERS                                       \
				EAST + NORTH + TOP, EAST + SOUTH + TOP,       \
				WEST + NORTH + TOP, WEST + SOUTH + TOP,       \
				EAST + NORTH + BOTTOM, EAST + SOUTH + BOTTOM, \
				WEST + NORTH + BOTTOM, WEST + SOUTH + BOTTOM


		static constexpr D straight[] = { STRAIGHT } ;
		static constexpr D slanting[] = { SLANTING } ;
		static constexpr D D3Q19[] = { STRAIGHT, SLANTING } ;
		static constexpr D D3Q27[] = { STRAIGHT, SLANTING, CORNERS } ;

#undef STRAIGHT
#undef SLANTING
#undef CORNERS

		// corresponds to constant 30 used by RS for marking nodes 
		// with more than 3 normal vectors
		static constexpr D RS_STRANGE_MARKING = 0b00101010 ; 

		HD Direction( D direction ) ;
		HD Direction() ;

		HD D get() const ;

		HD int getX() const ;
		HD int getY() const ;
		HD int getZ() const ;

		HD void setX( int value ) ;
		HD void setY( int value ) ;
		HD void setZ( int value ) ;

		HD Direction computeInverse() const ;

		typedef unsigned DirectionIndex ;

		// FIXME: remove, now this functionality is moved to LatticeArrangement specializations.
		HD DirectionIndex getIndexD3Q27() const ;
	
		//TODO: Does not check, if addition result has coordinates in allowed range.
		HD Direction operator+( const Direction & direction ) const ;
		//TODO: Does not check, if addition has coordinates in allowed range.
		HD Direction operator-( const Direction & direction ) const ;
		HD bool operator==( const Direction & direction ) const ;
		HD bool operator!=( const Direction & direction ) const ;
} ;



//TODO: Does not check, if result has coordinates in allowed range.
HD Direction crossProduct (const Direction & d1, const Direction & d2) ;
HD int dotProduct (const Direction & d1, const Direction & d2) ;



std::ostream& operator<<(std::ostream& out, const Direction & direction) ;
HD const char * toString( const Direction & direction ) ;


constexpr Direction::D   SELF = Direction::SELF ;
constexpr Direction::D   O    = Direction::SELF ;

constexpr Direction::D   N   = Direction::NORTH ;
constexpr Direction::D   S   = Direction::SOUTH ;
constexpr Direction::D   E   = Direction::EAST ;
constexpr Direction::D   W   = Direction::WEST ;
constexpr Direction::D   T   = Direction::TOP ;
constexpr Direction::D   B   = Direction::BOTTOM ;

constexpr Direction::D   NE  = Direction::NORTH + Direction::EAST ;
constexpr Direction::D   NW  = Direction::NORTH + Direction::WEST ;
constexpr Direction::D   NT  = Direction::NORTH + Direction::TOP ;
constexpr Direction::D   NB  = Direction::NORTH + Direction::BOTTOM ;

constexpr Direction::D   SE  = Direction::SOUTH + Direction::EAST ;
constexpr Direction::D   SW  = Direction::SOUTH + Direction::WEST ;
constexpr Direction::D   ST  = Direction::SOUTH + Direction::TOP ;
constexpr Direction::D   SB  = Direction::SOUTH + Direction::BOTTOM ;

constexpr Direction::D   ET  = Direction::EAST + Direction::TOP ;
constexpr Direction::D   EB  = Direction::EAST + Direction::BOTTOM ;

constexpr Direction::D   WT  = Direction::WEST + Direction::TOP ;
constexpr Direction::D   WB  = Direction::WEST + Direction::BOTTOM ;

constexpr Direction::D   NET = Direction::NORTH + Direction::EAST + Direction::TOP ;
constexpr Direction::D   NWT = Direction::NORTH + Direction::WEST + Direction::TOP ;
constexpr Direction::D   SET = Direction::SOUTH + Direction::EAST + Direction::TOP ;
constexpr Direction::D   SWT = Direction::SOUTH + Direction::WEST + Direction::TOP ;
constexpr Direction::D   NEB = Direction::NORTH + Direction::EAST + Direction::BOTTOM ;
constexpr Direction::D   NWB = Direction::NORTH + Direction::WEST + Direction::BOTTOM ;
constexpr Direction::D   SEB = Direction::SOUTH + Direction::EAST + Direction::BOTTOM ;
constexpr Direction::D   SWB = Direction::SOUTH + Direction::WEST + Direction::BOTTOM ;


// To keep the same computation order as in RS code we need to iterate over f_i
// using the same order.
Direction::D constexpr rsDirections[] = {O, E, N, W, S, T, B, NE, NW, SW, SE, ET, EB, WB, WT, NT, NB, SB, ST } ;


}



#include "Direction.hh"



#undef DIRECTION_TYPE



#endif
