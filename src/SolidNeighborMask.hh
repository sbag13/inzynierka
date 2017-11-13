#ifndef SOLID_NEIGHBOR_MASK_HH
#define SOLID_NEIGHBOR_MASK_HH



namespace microflow
{



inline
HD
SolidNeighborMask::
SolidNeighborMask()
{
	solidNeighborMask_.clear() ;
}



inline
void SolidNeighborMask::
markSolidNeighbor( const Direction & direction )
{
	Direction::DirectionIndex directionIndex = direction.getIndexD3Q27() ;

	markSolidNeighbor( directionIndex ) ;
}



inline
void SolidNeighborMask::
markSolidNeighbor( const Direction::DirectionIndex & directionIndex )
{
	unsigned bitIndex = directionIndex ;

	solidNeighborMask_.set( bitIndex ) ;
}



inline
HD
bool SolidNeighborMask::
isNeighborSolid( const Direction & direction ) const
{
	unsigned bitIndex = direction.getIndexD3Q27() ;

	return solidNeighborMask_.test( bitIndex ) ;
}



inline
bool SolidNeighborMask::
hasAllStraightNeighborsNonSolid() const
{
	for ( Direction d : Direction::straight )
	{
		if ( isNeighborSolid(d) )
			return false ;
	}
	return true ;
}



inline
bool SolidNeighborMask::
hasAllStraightAndSlantingNeighborsNonSolid() const
{
	if (not hasAllStraightNeighborsNonSolid() )
	{
		return false ;
	}

	for ( Direction d : Direction::slanting )
	{
		if ( isNeighborSolid(d) )
			return false ;
	}
	return true ;
}



inline
Direction SolidNeighborMask::
computeDirectionOfPlaneWithAllSolidNeighbors() const
{
	Direction solidPlaneDirection ;
	unsigned nSolidNeighbors = 0 ;

	for (Direction d : Direction::straight)
	{
		if ( isNeighborSolid(d) )
		{
			nSolidNeighbors ++ ;
			solidPlaneDirection = d ;
		}
	}

	if (1 != nSolidNeighbors)
	{
		return Direction() ;
	}

	SolidNeighborMask allowedSolidNeighbors ;

	allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection ) ;

	switch ( solidPlaneDirection.get() )
	{
		case Direction::EAST:
		case Direction::WEST:
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::TOP ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::BOTTOM ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::NORTH ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::SOUTH ) ;
			
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::TOP    +
																																		 Direction::NORTH ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::TOP    +
																																		 Direction::SOUTH ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::BOTTOM +
																																		 Direction::NORTH ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::BOTTOM +
																																		 Direction::SOUTH ) ;
		break ;

		case Direction::BOTTOM:
		case Direction::TOP   :
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::NORTH ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::SOUTH ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::EAST ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::WEST ) ;

			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::NORTH +
																																		 Direction::EAST ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::NORTH +
																																		 Direction::WEST ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::SOUTH +
																																		 Direction::EAST ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::SOUTH +
																																		 Direction::WEST ) ;
		break ;

		case Direction::NORTH:
		case Direction::SOUTH:
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::TOP ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::BOTTOM ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::EAST ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::WEST ) ;

			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::TOP +
																																		 Direction::EAST ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::TOP +
																																		 Direction::WEST ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::BOTTOM +
																																		 Direction::EAST ) ;
			allowedSolidNeighbors.markSolidNeighbor( solidPlaneDirection + Direction::BOTTOM +
																																		 Direction::WEST ) ;
	}


	if (	allowedSolidNeighbors.solidNeighborMask_ == 
				(
					solidNeighborMask_ | allowedSolidNeighbors.solidNeighborMask_
				)
		 )
	{
		return solidPlaneDirection ;
	}

	return Direction() ;
}




inline
std::ostream& operator<<(std::ostream& out, const SolidNeighborMask & solidNeighborMask)
{
	out << "0b" << std::hex << solidNeighborMask.solidNeighborMask_ ;

	return out ;
}



}



#endif
