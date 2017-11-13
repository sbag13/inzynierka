#ifndef LATTICE_ARRANGEMENT_D3Q27_HPP
#define LATTICE_ARRANGEMENT_D3Q27_HPP



#include "LatticeArrangement.hpp"



namespace microflow
{



template<>
class LatticeArrangement<3,27> : public LatticeArrangementBase<3,27>
{
	public:

		static constexpr Direction::D c[27] = { O,
																						E, N, W, S, T, B,
																						NE, SE, ET, EB, NW, SW, WT, WB, NT, NB, ST, SB,
																						NET, SET, NWT, SWT, NEB, SEB, NWB, SWB } ;

		// TODO: use CRTP
		HD static constexpr Direction::DirectionIndex getIndex( Direction::D direction )
		{
			return indexFromDirection_[ static_cast<unsigned>(direction) ] ;
		}
																								
	
	private:

		static constexpr Direction::DirectionIndex indexFromDirection_[ 0b00111111 + 1 ] =
		{
			       0,        1, NO_INDEX,        3,        2,    7 + 0, NO_INDEX,    7 + 4, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,        4,    7 + 1, NO_INDEX,    7 + 5, 
						 5,    7 + 2, NO_INDEX,    7 + 6,    7 + 8,   19 + 0, NO_INDEX,   19 + 2, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,   7 + 10,   19 + 1, NO_INDEX,   19 + 3, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, 
						 6,    7 + 3, NO_INDEX,    7 + 7,    7 + 9,   19 + 4, NO_INDEX,   19 + 6, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,   7 + 11,   19 + 5, NO_INDEX,   19 + 7 
		} ;
} ;



typedef LatticeArrangement<3,27>    D3Q27 ;



}



#endif
