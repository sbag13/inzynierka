#ifndef PACKED_NODE_NORMAL_SET_HH
#define PACKED_NODE_NORMAL_SET_HH



#include "Coordinates.hpp"



namespace microflow
{



INLINE
HD
PackedNodeNormalSet::
PackedNodeNormalSet()
: normalVectorsCounter_(0), edgeNodeType_(0), normalVectors_(0) 
{
}



INLINE
HD
Direction PackedNodeNormalSet::
getNormalVector( unsigned vectorIndex ) const
{
	PackedNodeNormalSet::VectorType packedVector ;

	packedVector = extractNormalVector( vectorIndex ) ;

	return Direction( packedVector ) ;
}



inline
__host__
void PackedNodeNormalSet::
addNormalVector( const Direction & normalVector )
{
	for (unsigned i=0 ; i < getNormalVectorsCounter() ; i++)
	{
		Direction localVector = getNormalVector(i) ;

		Direction::D packedLocalVector  = localVector.get() & 0b00111111 ;
		Direction::D packedNormalVector = normalVector.get() & 0b00111111 ;
		if ( packedLocalVector == packedNormalVector )
		{
			return ;
		}
	}

	if (getNormalVectorsCounter() >= 6)
	{
		THROW ("There are no free places in PackedNodeNormalSet") ;
	}

	PackedDirectionVector<long long> normalVectors( normalVectors_ ) ;

	unsigned newVectorPosition = getNormalVectorsCounter() ;
	normalVectors.setVector3D( newVectorPosition, normalVector ) ;

	normalVectorsCounter_ ++ ;
	normalVectors_ = normalVectors.getInternalStorage() ;
}



INLINE
HD
unsigned  PackedNodeNormalSet::
getNormalVectorsCounter() const 
{
	return normalVectorsCounter_ ;
}



INLINE
HD
Direction PackedNodeNormalSet::
getResultantNormalVector() const
{
	return getNormalVector(6) ;
}



INLINE
HD
enum PackedNodeNormalSet::EdgeNodeType PackedNodeNormalSet::
getEdgeNodeType() const
{
	return static_cast<PackedNodeNormalSet::EdgeNodeType>(edgeNodeType_) ;
}



INLINE
HD
void PackedNodeNormalSet::
setEdgeNodeType( enum PackedNodeNormalSet::EdgeNodeType edgeNodeType )
{
	edgeNodeType_ = edgeNodeType ;
}



inline
void PackedNodeNormalSet::
calculateResultantNormalVector()
{
	Direction resultantNormalVectorPacked(0) ;

	if (getNormalVectorsCounter() < 4)
	{
		Coordinates resultantNormalVector(0) ;

		for (size_t i=0 ; i < getNormalVectorsCounter() ; i++)
		{
			resultantNormalVector = resultantNormalVector + getNormalVector(i) ;
		}

		resultantNormalVectorPacked.setX( resultantNormalVector.getX() ) ;
		resultantNormalVectorPacked.setY( resultantNormalVector.getY() ) ;
		resultantNormalVectorPacked.setZ( resultantNormalVector.getZ() ) ;
	}
	else
	{
		resultantNormalVectorPacked = Direction(Direction::RS_STRANGE_MARKING) ;
	}

	PackedDirectionVector<long long> normalVectors( normalVectors_ ) ;
	normalVectors.setVector3D( 6, resultantNormalVectorPacked ) ;
	normalVectors_ = normalVectors.getInternalStorage() ;
}



INLINE
HD
PackedNodeNormalSet::VectorType PackedNodeNormalSet::
extractNormalVector( unsigned vectorIndex ) const
{
	ASSERT( 7 > vectorIndex ) ;

	PackedDirectionVector<long long> packedVectors( normalVectors_ ) ;

	return packedVectors.getVector3D( vectorIndex ) ;
}


inline
std::ostream & 
operator<<(std::ostream& out, const PackedNodeNormalSet & packedNodeNormalSet)
{
	unsigned nNormalVectors = packedNodeNormalSet.getNormalVectorsCounter() ;

	out << "{" ;

	out << nNormalVectors << " normal vectors" ;
	if (0 < nNormalVectors)
	{
		out << ":" ;
	}
	for (unsigned i=0 ; i < nNormalVectors ; i++)
	{
		Direction normalVector = packedNodeNormalSet.getNormalVector(i) ;
		out << " (" ;
		out << normalVector.getX() << "," ;
		out << normalVector.getY() << "," ;
		out << normalVector.getZ() ;
		out << ")" ;
	}
	
	out << ", edgeNodeType=" << packedNodeNormalSet.getEdgeNodeType() ;

	out << ", resultantNormalVector:" ;
	out << " (" ;
	out << packedNodeNormalSet.getResultantNormalVector().getX() << "," ;
	out << packedNodeNormalSet.getResultantNormalVector().getY() << "," ;
	out << packedNodeNormalSet.getResultantNormalVector().getZ() ;
	out << ")" ;

	out << "}" ;
	return out ;
}



}
#endif
