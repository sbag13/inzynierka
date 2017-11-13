#ifndef TILED_LATTICE_HH
#define TILED_LATTICE_HH



#include "Exceptions.hpp"



namespace microflow
{



inline
TiledLatticeBaseTwoCopies::
TiledLatticeBaseTwoCopies()
: validCopyID_ (ValidCopyID::NONE)
{
}



inline 
TiledLatticeBaseTwoCopies::ValidCopyID TiledLatticeBaseTwoCopies::
getValidCopyID() const
{
	return validCopyID_ ;
}



inline 
bool TiledLatticeBaseTwoCopies::
isValidCopyID (ValidCopyID validCopyID) const
{
	return (validCopyID == validCopyID_) ;
}



inline 
bool TiledLatticeBaseTwoCopies::
isValidCopyIDNone () const
{
	return isValidCopyID (ValidCopyID::NONE) ;
}



inline 
bool TiledLatticeBaseTwoCopies::
isValidCopyIDF () const
{
	return isValidCopyID (ValidCopyID::F) ;
}



inline 
bool TiledLatticeBaseTwoCopies::
isValidCopyIDFPost () const
{
	return isValidCopyID (ValidCopyID::FPOST) ;
}



inline 
void TiledLatticeBaseTwoCopies::
setValidCopyID (ValidCopyID validCopyID)
{
	validCopyID_ = validCopyID ;
}



inline 
void TiledLatticeBaseTwoCopies::
setValidCopyIDToNone()
{
	setValidCopyID (ValidCopyID::NONE) ;
}



inline 
void TiledLatticeBaseTwoCopies::
setValidCopyIDToF()
{
	setValidCopyID (ValidCopyID::F) ;
}



inline 
void TiledLatticeBaseTwoCopies::
setValidCopyIDToFPost()
{
	setValidCopyID (ValidCopyID::FPOST) ;
}



inline
void TiledLatticeBaseTwoCopies::
switchValidCopyID()
{
	switch (getValidCopyID())
	{
		case ValidCopyID::F:
			setValidCopyIDToFPost() ;
			break ;

		case ValidCopyID::FPOST:
			setValidCopyIDToF() ;
			break ;

		case ValidCopyID::NONE:
			break ;
	}
}



#define TEMPLATE_TILED_LATTICE_NO_INLINE       \
template<class LatticeArrangement, class DataType, TileDataArrangement DataArrangement >



#define TEMPLATE_TILED_LATTICE   \
TEMPLATE_TILED_LATTICE_NO_INLINE \
inline



#define TILED_LATTICE   \
TiledLattice< LatticeArrangement, DataType, StorageOnCPU, DataArrangement >



TEMPLATE_TILED_LATTICE
const typename TILED_LATTICE::TileType
TILED_LATTICE::
getTile( ConstIterator tileIndex ) const
{
	if (tileIndex >= getEndOfTiles())
	{
		THROW ("tileIndex too large") ;
	}

	return
		TileType
			( tileIndex, tileLayout_, 
			// FIXME: check, whether below cons_casts are safe
			const_cast< StorageOnCPU<NodeType> & >(nodeTypes_), 
			const_cast< StorageOnCPU<PackedNodeNormalSet> & >(nodeNormals_), 
			const_cast< StorageOnCPU<SolidNeighborMask> & >(solidNeighborMasks_), 
			const_cast< StorageOnCPU<DataType> & >(allValues_ ) ) ;
}



TEMPLATE_TILED_LATTICE
typename TILED_LATTICE::TileType
TILED_LATTICE::
getTile( Iterator tileIndex )
{
	if (tileIndex >= getEndOfTiles())
	{
		THROW ("tileIndex too large") ;
	}

	return
		TileType
			( tileIndex, tileLayout_, nodeTypes_, nodeNormals_, solidNeighborMasks_, allValues_ ) ;
}



TEMPLATE_TILED_LATTICE
unsigned TILED_LATTICE::
getNOfTiles() const
{
	return tileLayout_.getNoNonEmptyTiles() ;
}



TEMPLATE_TILED_LATTICE
typename TILED_LATTICE::Iterator
TILED_LATTICE::
getBeginOfTiles()
{
	return tileLayout_.getBeginOfNonEmptyTiles() ;
}



TEMPLATE_TILED_LATTICE
typename TILED_LATTICE::Iterator
TILED_LATTICE::
getEndOfTiles()
{
	return tileLayout_.getEndOfNonEmptyTiles() ;
}



TEMPLATE_TILED_LATTICE
typename TILED_LATTICE::ConstIterator
TILED_LATTICE::
getBeginOfTiles() const
{
	return tileLayout_.getBeginOfNonEmptyTiles() ;
}



TEMPLATE_TILED_LATTICE
typename TILED_LATTICE::ConstIterator
TILED_LATTICE::
getEndOfTiles() const
{
	return tileLayout_.getEndOfNonEmptyTiles() ;
}



TEMPLATE_TILED_LATTICE
TileLayout<StorageOnCPU> &
TILED_LATTICE::
getTileLayout() 
{
	return tileLayout_ ;
}



TEMPLATE_TILED_LATTICE
const TileLayout<StorageOnCPU> &
TILED_LATTICE::
getTileLayout() const
{
	return tileLayout_ ;
}



TEMPLATE_TILED_LATTICE
bool TILED_LATTICE::
operator==( const TILED_LATTICE & tiledLattice ) const
{
	if (
			tiledLattice.nodeTypes_ == nodeTypes_ &&
			tiledLattice.allValues_ == allValues_
		 )
	{
		return true ;
	}
	return false ;
}



TEMPLATE_TILED_LATTICE_NO_INLINE
template <class Functor>
inline
void TILED_LATTICE::
forEachTile (Functor functor)
{
	for (auto t = getBeginOfTiles() ; 
						t < getEndOfTiles() ; 
						t++)
	{
		auto tile = getTile (t) ;
		functor (tile) ;
	}
}



TEMPLATE_TILED_LATTICE_NO_INLINE
template <class Functor>
inline
void TILED_LATTICE::
forEachNode (Functor functor)
{
	forEachTile
	(
		[&] (TileType & tile)
		{
			Coordinates tileCorner = tile.getCornerPosition() ;
		
			constexpr unsigned edge = getNNodesPerTileEdge() ;
			for (unsigned tz=0 ; tz < edge ; tz++)
				for (unsigned ty=0 ; ty < edge ; ty++)
					for (unsigned tx=0 ; tx < edge ; tx++)
					{
						auto node = tile.getNode (tx,ty,tz) ;
						Coordinates globalCoordinates = tileCorner + Coordinates (tx,ty,tz) ;
		
						functor (node, globalCoordinates) ;
					}
		}
	) ;
}



#undef TEMPLATE_TILED_LATTICE
#undef TILED_LATTICE



}



#endif
