#ifndef NODE_FROM_TILE_HH
#define NODE_FROM_TILE_HH



#include "cudaPrefix.hpp"



namespace microflow
{



/*
		NodeFromTile <REFERENCE>
*/



#define NODE_FROM_TILE_REFERENCE   \
NodeFromTile <Tile, DataStorageMethod::REFERENCE>



template <class Tile>
INLINE
NODE_FROM_TILE_REFERENCE::
NodeFromTile (Tile & tile, unsigned x, unsigned y, unsigned z)
: x_(x), y_(y), z_(z), tile_(tile)
{
}



template <class Tile>
INLINE
NodeType & NODE_FROM_TILE_REFERENCE::
nodeType()
{
	return tile_.getNodeTypes()[z_][y_][x_] ;
}



template <class Tile>
INLINE
PackedNodeNormalSet & NODE_FROM_TILE_REFERENCE::
nodeNormals()
{
	return tile_.getNodeNormals()[z_][y_][x_] ;
}



template <class Tile>
INLINE
SolidNeighborMask & NODE_FROM_TILE_REFERENCE::
solidNeighborMask()
{
	return tile_.getNodeSolidNeighborMasks()[z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
f( Direction direction )
{
	return tile_.getFPtr (direction) 
				 [
				 	Tile::computeIndexInFArray 
						(
							x_,y_,z_, 
							Tile::LatticeArrangementType::getIndex (direction.get())
						)
				 ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
f( Direction::D direction )
{
	return f( Direction(direction) ) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
f( Direction::DirectionIndex index )
{
	return f (Tile::LatticeArrangementType::getC(index)) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
fPost( Direction direction )
{
	return tile_.getFPostPtr (direction) 
				 [
				 	Tile::computeIndexInFArray 
						(
							x_,y_,z_, 
							Tile::LatticeArrangementType::getIndex (direction.get())
						)
				 ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
fPost( Direction::D direction )
{
	return fPost( Direction(direction) ) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
fPost( Direction::DirectionIndex index )
{
	return fPost (Tile::LatticeArrangementType::getC(index)) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
rho()
{
	return tile_.rho()[z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
rho0()
{
	return tile_.rho0()[z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
rhoBoundary()
{
	return tile_.rhoBoundary()[z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
u( Axis axis )
{
	return tile_.u( axis )[z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
uBoundary( Axis axis )
{
	return tile_.uBoundary( axis )[z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
uT0( Axis axis )
{
	return tile_.uT0( axis )[z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
u( unsigned axis )
{
	return tile_.u()[axis][z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
uBoundary( unsigned axis )
{
	return tile_.uBoundary()[axis][z_][y_][x_] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_REFERENCE::DataTypeType & NODE_FROM_TILE_REFERENCE::
uT0( unsigned axis )
{
	return tile_.uT0()[axis][z_][y_][x_] ;
}



#undef NODE_FROM_TILE_REFERENCE



/*
	NodeFromTile <POINTERS>
*/



#define NODE_FROM_TILE_POINTERS   \
NodeFromTile <Tile, DataStorageMethod::POINTERS>



template <class Tile>
INLINE
unsigned NODE_FROM_TILE_POINTERS::
getTileIndex() const
{
	return tileIndex_ ;
}



template <class Tile>
INLINE
NODE_FROM_TILE_POINTERS::
NodeFromTile
(
 unsigned x, unsigned y, unsigned z, unsigned tileIndex,
 NodeType          * tiledNodeTypes,
 SolidNeighborMask * tiledSolidNeighborMasks,
 PackedNodeNormalSet * tiledNodeNormals,
 DataTypeType        * tiledAllValues
 )
: x_ (x), y_ (y), z_ (z), tileIndex_ (tileIndex),
	tiledNodeTypes_ (tiledNodeTypes),
	tiledSolidNeighborMasks_ (tiledSolidNeighborMasks),
	tiledNodeNormals_ (tiledNodeNormals),
	tiledAllValues_ (tiledAllValues)
{}



template <class Tile>
INLINE 
NodeType & 
NODE_FROM_TILE_POINTERS::
nodeType()
{
	return tiledNodeTypes_ [computeNodeIndex()] ;
}



template <class Tile>
INLINE
SolidNeighborMask & 
NODE_FROM_TILE_POINTERS::
solidNeighborMask()
{
	return tiledSolidNeighborMasks_ [computeNodeIndex()] ;
}



template <class Tile>
INLINE
PackedNodeNormalSet & 
NODE_FROM_TILE_POINTERS::
nodeNormals()
{
	return tiledNodeNormals_ [computeNodeIndex()] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
u (unsigned axis)
{
	return u (Axis(axis)) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType &
NODE_FROM_TILE_POINTERS::
u (Axis axis)
{
	return tiledAllValues_ [ computeNodeDataIndex (Tile::Data::U, axis) ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
uT0 (unsigned axis)
{
	return uT0 (Axis(axis)) ;
}

template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType &
NODE_FROM_TILE_POINTERS::
uT0 (Axis axis)
{
	return tiledAllValues_ [ computeNodeDataIndex (Tile::Data::U_T0, axis) ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
uBoundary (unsigned axis)
{
	return uBoundary (Axis(axis)) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType &
NODE_FROM_TILE_POINTERS::
uBoundary (Axis axis)
{
	return tiledAllValues_ [ computeNodeDataIndex (Tile::Data::U_BOUNDARY, axis) ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
rho()
{
	return tiledAllValues_ [ computeNodeDataIndex (Tile::Data::RHO) ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
rho0()
{
	return tiledAllValues_ [ computeNodeDataIndex (Tile::Data::RHO_T0) ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
rhoBoundary()
{
	return tiledAllValues_ [ computeNodeDataIndex (Tile::Data::RHO_BOUNDARY) ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
f (Direction::DirectionIndex index)
{
	return tiledAllValues_ [ computeNodeDataIndex (Tile::Data::F, index) ] ;
}

template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
f (Direction direction)
{
	const Direction::DirectionIndex 
		directionIndex = Tile::LatticeArrangementType::getIndex( direction.get() ) ;

	return f (directionIndex) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
f (Direction::D direction)
{
	return f (Direction(direction)) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
fPost (Direction::DirectionIndex index)
{
	return tiledAllValues_ [ computeNodeDataIndex (Tile::Data::F_POST, index) ] ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
fPost (Direction direction)
{
	const Direction::DirectionIndex 
		directionIndex = Tile::LatticeArrangementType::getIndex( direction.get() ) ;

	return fPost (directionIndex) ;
}



template <class Tile>
INLINE
typename NODE_FROM_TILE_POINTERS::DataTypeType & 
NODE_FROM_TILE_POINTERS::
fPost (Direction::D direction)
{
	return fPost (Direction(direction)) ;
}



template <class Tile>
INLINE 
unsigned 
NODE_FROM_TILE_POINTERS::
computeNodeIndex() const
{
	return Tile::TraitsType::computeNodeIndex (x_, y_, z_, tileIndex_) ;
}



template <class Tile>
INLINE
unsigned 
NODE_FROM_TILE_POINTERS::
computeNodeDataIndex (typename Tile::Data data, Axis axis)
{
	return Tile::computeNodeDataIndex (x_, y_, z_, tileIndex_, data, axis) ;
}



template <class Tile>
INLINE
unsigned 
NODE_FROM_TILE_POINTERS::
computeNodeDataIndex (typename Tile::Data data, Direction::DirectionIndex fIndex)
{
	return Tile::computeNodeDataIndex (x_, y_, z_, tileIndex_, data, fIndex) ;
}



#undef NODE_FROM_TILE_POINTERS



/*
	NodeFromTile <COPY>
*/



#define NODE_FROM_TILE_COPY   \
NodeFromTile <Tile, DataStorageMethod::COPY>



template <class Tile>
INLINE
NODE_FROM_TILE_COPY::
NodeFromTile
(
	unsigned x, unsigned y, unsigned z, unsigned tileIndex,
	NodeType            * tiledNodeTypes,
	SolidNeighborMask   * tiledSolidNeighborMasks,
	PackedNodeNormalSet * tiledNodeNormals,
	typename NodeFromTile <Tile, DataStorageMethod::POINTERS>::DataTypeType * tiledAllValues
) 
: NodeFromTile <Tile, DataStorageMethod::POINTERS> 
(
	x,y,z, tileIndex,
	tiledNodeTypes, tiledSolidNeighborMasks, tiledNodeNormals, tiledAllValues
)
{}



template <class Tile>
INLINE DEVICE
void NODE_FROM_TILE_COPY::
registerSharedU 
(
	typename Tile::DataTypeType (&uGPU) [Tile::LatticeArrangementType::getD()][4][4][4]
)
{
	uGPU_ = & uGPU ;
}



template <class Tile>
INLINE
NodeType & NODE_FROM_TILE_COPY::
nodeType()
{
	return nodeType_ ;
}



template <class Tile>
INLINE
typename Tile::DataTypeType & NODE_FROM_TILE_COPY::
u (unsigned axis)
{ 
#ifndef __CUDA_ARCH__
	return u_ [axis] ; 
#else
	return (*uGPU_) [axis] [threadIdx.z][threadIdx.y][threadIdx.x] ;
#endif
}



template <class Tile>
INLINE
typename Tile::DataTypeType & NODE_FROM_TILE_COPY::
u (Axis axis)
{
	return u (static_cast<unsigned>(axis)) ;
}



template <class Tile>
INLINE
typename Tile::DataTypeType & NODE_FROM_TILE_COPY::
rho()
{
	return rho_ ;
}



template <class Tile>
INLINE
typename Tile::DataTypeType & NODE_FROM_TILE_COPY::
f (Direction::DirectionIndex index)
{
	return f_ [index] ;
}



template <class Tile>
INLINE
typename Tile::DataTypeType & NODE_FROM_TILE_COPY::
f (Direction direction)
{
	const Direction::DirectionIndex 
		directionIndex = Tile::LatticeArrangementType::getIndex( direction.get() ) ;

	return f (directionIndex) ;
}



template <class Tile>
INLINE
typename Tile::DataTypeType & NODE_FROM_TILE_COPY::
f (Direction::D direction)
{
	return f (Direction (direction)) ;
}



#undef NODE_FROM_TILE_COPY



}



#endif
