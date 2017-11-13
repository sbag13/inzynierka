#ifndef LATTICE_ARRANGEMENT_HPP
#define LATTICE_ARRANGEMENT_HPP



#include <string>



#include "Direction.hpp"
#include "cudaPrefix.hpp"



namespace microflow
{



template<unsigned D, unsigned Q>
class LatticeArrangementBase
{
	public:
		HD static constexpr unsigned getD() ;
		HD static constexpr unsigned getQ() ;

		HD static constexpr Direction::DirectionIndex getIndex( Direction::D direction) ;

		static const std::string getName() ;

	protected:
		// Should result in sgmentation fault
		static constexpr Direction::DirectionIndex 
							NO_INDEX = std::numeric_limits<Direction::DirectionIndex>::max() ;

	private:
		static constexpr Direction::DirectionIndex indexFromDirection_[ 0b00111111 + 1 ] =
		{
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX,
			NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX, NO_INDEX
		} ;
} ;



template<unsigned D, unsigned Q>
class LatticeArrangement : public LatticeArrangementBase<D,Q>
{
} ;



}



#include "LatticeArrangement.hh"

#include "LatticeArrangementD3Q19.hpp"
#include "LatticeArrangementD3Q27.hpp"



#endif
