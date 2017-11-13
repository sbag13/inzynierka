#ifndef LATTICE_ARRANGEMENT_D3Q19_HPP
#define LATTICE_ARRANGEMENT_D3Q19_HPP



#include "LatticeArrangement.hpp"



namespace microflow
{



template<>
class LatticeArrangement<3,19> : public LatticeArrangementBase<3,19>
{
	public:
		static constexpr Direction::D c[19] = { O,
																						E, N, W, S, T, B,
																						NE, SE, ET, EB, NW, SW, WT, WB, NT, NB, ST, SB
																					} ;

		// TODO: DataType instead of double
		static constexpr double w[19] = { 1.0/3.0,
																			1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,
																			1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,
																			1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0
																		} ;
		static constexpr double csq = 1.0 / 3.0 ;

		HD static CONSTEXPR
		Direction::DirectionIndex getIndex( Direction::D direction ) ;

		HD static CONSTEXPR
		Direction::D getC( Direction::DirectionIndex index ) ;

		HD static CONSTEXPR
		double getW( Direction::DirectionIndex index ) ;


	private:
		static constexpr Direction::DirectionIndex indexFromDirection_[ 0b00111111 + 1 ] =
		{
			       0,        1, NO_INDEX,        3,        2,    7 + 0, NO_INDEX,    7 + 4, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,        4,    7 + 1, NO_INDEX,    7 + 5, 
						 5,    7 + 2, NO_INDEX,    7 + 6,    7 + 8, NO_INDEX, NO_INDEX, NO_INDEX, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,   7 + 10, NO_INDEX, NO_INDEX, NO_INDEX, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, 
						 6,    7 + 3, NO_INDEX,    7 + 7,    7 + 9, NO_INDEX, NO_INDEX, NO_INDEX, 
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,   7 + 11, NO_INDEX, NO_INDEX, NO_INDEX 
		} ;
		
} ;



typedef LatticeArrangement<3,19>    D3Q19 ;



}



#include "LatticeArrangementD3Q19.hh"



#endif
