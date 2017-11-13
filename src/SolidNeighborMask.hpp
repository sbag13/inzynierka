#ifndef SOLID_NEIGHBOR_MASK_HPP
#define SOLID_NEIGHBOR_MASK_HPP



#include <bitset>
#include <ostream>

#include "Direction.hpp"
#include "BitSet.hpp"



namespace microflow
{



class SolidNeighborMask
{
	public:

		HD SolidNeighborMask() ;

		void markSolidNeighbor( const Direction & direction ) ;
		// FIXME: in method below normalize the values of directionIndex - now it must
		//        be consistent with the order in Direction::D3Q27 array.
		void markSolidNeighbor( const Direction::DirectionIndex & directionIndex ) ;
		HD bool isNeighborSolid( const Direction & direction ) const ;

		bool hasAllStraightNeighborsNonSolid() const ;
		bool hasAllStraightAndSlantingNeighborsNonSolid() const ;
		bool hasSolidNeighbor() const { return not solidNeighborMask_.isClear() ; }

		// Returns empty Direction if solid neighbors are not on single plane.
		Direction computeDirectionOfPlaneWithAllSolidNeighbors() const ;

	private:
		BitSet solidNeighborMask_ ; // TODO: BitSet<29>

	friend
	std::ostream& operator<<(std::ostream& out, const SolidNeighborMask & solidNeighborMask) ;
} ;



}



#include "SolidNeighborMask.hh"



#endif
