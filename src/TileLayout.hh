#ifndef TILE_LAYOUT_HH
#define TILE_LAYOUT_HH



#include "TileDefinitions.hpp"
#include "Exceptions.hpp"



namespace microflow
{



template< template<class> class Storage >
INLINE
TileLayoutBase<Storage>::NonEmptyTile::
NonEmptyTile( TileLayoutBase & tileLayoutBase, TileIndex tileIndex )
: tileLayoutBase_( tileLayoutBase ), 
	tileIndex_( tileIndex )
{
}



template< template<class> class Storage >
INLINE
Coordinates TileLayoutBase<Storage>::NonEmptyTile::
getCornerPosition() const
{
	return tileLayoutBase_.getNonEmptyTilePosition( tileIndex_ ) ;
}



template< template<class> class Storage >
INLINE
Coordinates TileLayoutBase<Storage>::NonEmptyTile::
getMapPosition() const
{
	Coordinates cornerPosition = getCornerPosition() ;

	const size_t tileEdge = TileLayoutBase<Storage>::NonEmptyTile::getNNodesPerEdge() ;
	
	Coordinates mapPosition( 
						cornerPosition.getX() / tileEdge,  
						cornerPosition.getY() / tileEdge,  
						cornerPosition.getZ() / tileEdge
						) ;

	return mapPosition ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::NonEmptyTile::Iterator TileLayoutBase<Storage>::NonEmptyTile::
getBeginOfNodes() const
{
	return 0 ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::NonEmptyTile::Iterator TileLayoutBase<Storage>::NonEmptyTile::
getEndOfNodes() const
{
	return getNNodesPerTile() ;
}



template< template<class> class Storage >
INLINE
Coordinates TileLayoutBase<Storage>::NonEmptyTile::
unpack( TileLayoutBase<Storage>::NonEmptyTile::Iterator iterator ) const
{
	// Unpack linear iterator into x,y and z offsets
	size_t offsetX = iterator % DEFAULT_3D_TILE_EDGE ;
	size_t offsetY = (iterator / DEFAULT_3D_TILE_EDGE) % DEFAULT_3D_TILE_EDGE ; 
	size_t offsetZ = iterator / (DEFAULT_3D_TILE_EDGE * DEFAULT_3D_TILE_EDGE) ;

	return Coordinates( offsetX, offsetY, offsetZ ) + getCornerPosition() ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::NonEmptyTile::TileIndex TileLayoutBase<Storage>::NonEmptyTile::
getIndex() const
{
	return tileIndex_ ;
}



template< template<class> class Storage >
template< template<class> class InputStorage >
INLINE
TileLayoutBase< Storage >::
TileLayoutBase( 
		InputStorage<size_t> & tilesX0,
		InputStorage<size_t> & tilesY0,
		InputStorage<size_t> & tilesZ0,
		LinearizedMatrix< unsigned int, InputStorage > & tileMap,
		Size & size
		) 
: tilesX0_( tilesX0 ),
	tilesY0_( tilesY0 ),
	tilesZ0_( tilesZ0 ),
	tileMap_( tileMap ),
	size_   ( size    )
{
}


template< template<class> class Storage >
INLINE
Size TileLayoutBase<Storage>::
getSize() const
{
	return size_ ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::ConstIterator TileLayoutBase<Storage>::
getBeginOfNonEmptyTiles() const
{
	return 0 ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::Iterator TileLayoutBase<Storage>::
getBeginOfNonEmptyTiles()
{
	return 0 ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::ConstIterator TileLayoutBase<Storage>::
getEndOfNonEmptyTiles() const
{
	return getNoNonEmptyTiles() ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::Iterator TileLayoutBase<Storage>::
getEndOfNonEmptyTiles()
{
	return getNoNonEmptyTiles() ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::NonEmptyTile TileLayoutBase<Storage>::
getTile( TileLayoutBase<Storage>::Iterator it )
{
	ASSERT( it >= getBeginOfNonEmptyTiles() ) ;
	ASSERT( it <  getEndOfNonEmptyTiles() ) ;

	return TileLayoutBase<Storage>::NonEmptyTile( *this, it ) ;
}



template< template<class> class Storage >
INLINE
const typename TileLayoutBase<Storage>::NonEmptyTile TileLayoutBase<Storage>::
getTile( TileLayoutBase<Storage>::ConstIterator it ) const
{
	ASSERT( it >= getBeginOfNonEmptyTiles() ) ;
	ASSERT( it <  getEndOfNonEmptyTiles() ) ;

	// const_cast, because returned value is const
	return TileLayoutBase<Storage>::NonEmptyTile( const_cast<TileLayoutBase<Storage> &>(*this), it ) ;
}



template< template<class> class Storage >
INLINE
const typename TileLayoutBase<Storage>::NonEmptyTile TileLayoutBase<Storage>::
getTile( Coordinates nodeCoordinates ) const
{
	unsigned nx = nodeCoordinates.getX() ;
	unsigned ny = nodeCoordinates.getY() ;
	unsigned nz = nodeCoordinates.getZ() ;

	Coordinates tileCorner( 
													nx - (nx % DEFAULT_3D_TILE_EDGE),
													ny - (ny % DEFAULT_3D_TILE_EDGE),
													nz - (nz % DEFAULT_3D_TILE_EDGE)
												) ;

	Coordinates tileMapPosition(
													tileCorner.getX() / DEFAULT_3D_TILE_EDGE,
													tileCorner.getY() / DEFAULT_3D_TILE_EDGE,
													tileCorner.getZ() / DEFAULT_3D_TILE_EDGE
														 ) ;

	if ( not getSize().areCoordinatesInLimits( tileMapPosition ) )
	{
		THROW ("Node coordinates are outside geometry") ;
	}
	unsigned int tile = tileMap_.getValue( tileMapPosition ) ;
	if ( (EMPTY_TILE == tile) )
	{
		THROW ("Node coordinates are in empty tile") ;
	}

	return getTile( tile ) ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::ConstIterator TileLayoutBase<Storage>::
getNeighborIndex( Iterator currentTile, Direction direction ) const
{
	auto const tile = getTile( currentTile ) ;

	if ( hasNeighbour(tile, direction) )
	{
		auto const neighbor = getNeighbour(tile, direction) ;
		return neighbor.getIndex() ;
	}
	else
	{
		return EMPTY_TILE ;
	}
}



template< template<class> class Storage >
INLINE
bool TileLayoutBase<Storage>::
hasNeighbour( const TileLayoutBase<Storage>::NonEmptyTile & tile, Direction direction ) const
{
	Coordinates position = tile.getMapPosition() + direction ;

	if ( not getSize().areCoordinatesInLimits(position) )
	{
		return false ;
	}

	unsigned int neighbourTile = tileMap_.getValue( position ) ;

	if ( (EMPTY_TILE == neighbourTile) )
	{
		return false ;
	}

	return true ;
}



template< template<class> class Storage >
INLINE
const typename TileLayoutBase<Storage>::NonEmptyTile TileLayoutBase<Storage>::
getNeighbour( const TileLayoutBase<Storage>::NonEmptyTile & tile, Direction direction ) const
{
	if ( hasNeighbour( tile, direction ) )
	{
		Coordinates position = tile.getMapPosition() + direction ;
		unsigned int neighbourTile = tileMap_.getValue( position ) ;

		// const_cast seems safe, because the returned value is const
		return NonEmptyTile( const_cast<TileLayoutBase&>(*this), neighbourTile ) ;
	}

	THROW ("getNeighbour() called for empty tile") ;

	//FIXME: UNSAFE, used only to disable compilation warning ;
	return NonEmptyTile( const_cast<TileLayoutBase&>(*this), 0u-1u ) ;
}



template< template<class> class Storage >
INLINE
typename TileLayoutBase<Storage>::NonEmptyTile TileLayoutBase<Storage>::
getNeighbour( const TileLayoutBase<Storage>::NonEmptyTile & tile, Direction direction )
{
	const TileLayoutBase & constThis = *this ;

	// TODO: may be slow (copy contructor ???)
	return constThis.getNeighbour(tile, direction) ;
}



template< template<class> class Storage >
INLINE
size_t TileLayoutBase<Storage>::
getNoNonEmptyTiles() const
{
	return tilesX0_.size() ;
}



template< template<class> class Storage >
INLINE
Coordinates TileLayoutBase<Storage>::
getNonEmptyTilePosition( typename TileLayoutBase::NonEmptyTile::TileIndex tileIndex ) const
{
	return Coordinates( tilesX0_[tileIndex], tilesY0_[tileIndex], tilesZ0_[tileIndex] ) ;
}



inline
const NodeType & TileLayout< StorageOnCPU >::
getNodeType( Coordinates coordinates ) const
{
	return nodeLayout_.getNodeType( coordinates ) ;
}



inline
const NodeLayout & TileLayout<StorageOnCPU>::
getNodeLayout() const
{
	return nodeLayout_ ;
}



inline
const LinearizedMatrix<unsigned,StorageOnCPU> & TileLayout<StorageOnCPU>::
getTileMap() const
{
	return tileMap_ ;
}



inline
bool TileLayout<StorageOnCPU>::
operator==( const TileLayout<StorageOnCPU> & tileLayout ) const
{
	if (
			tileLayout.tilesX0_ == tilesX0_  &&
			tileLayout.tilesY0_ == tilesY0_  &&
			tileLayout.tilesZ0_ == tilesZ0_  &&

			tileLayout.tileMap_ == tileMap_  &&
			tileLayout.size_    == size_ 
		)
	{
		return true ;
	}

	return false ;
}



#undef TEMPLATE_TILE_LAYOUT
#undef TILE_LAYOUT_CPU



}



#endif
