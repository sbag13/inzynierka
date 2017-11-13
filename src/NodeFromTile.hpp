#ifndef NODE_FROM_TILE_HPP
#define NODE_FROM_TILE_HPP



#include "NodeType.hpp"
#include "PackedNodeNormalSet.hpp"
#include "SolidNeighborMask.hpp"
#include "Axis.hpp"

#include "LatticeArrangement.hpp" //FIXME: Required only for class NodeData.



namespace microflow
{



enum DataStorageMethod
{
	REFERENCE,
	POINTERS,
	COPY
} ;



template <class Tile, DataStorageMethod DataStorage>
class NodeFromTile ;



#define NODE_INTERFACE                                                  \
																																				\
	typedef typename Tile::LatticeArrangementType LatticeArrangementType ;\
                                                                        \
	HD NodeType & nodeType() ;                                            \
	HD PackedNodeNormalSet & nodeNormals() ;                              \
	HD SolidNeighborMask & solidNeighborMask() ;                          \
                                                                        \
	typedef typename Tile::DataTypeType DataTypeType ;                    \
                                                                        \
	HD DataTypeType & f    ( Direction                 direction ) ;      \
	HD DataTypeType & f    ( Direction::D              direction ) ;      \
	HD DataTypeType & f    ( Direction::DirectionIndex index ) ;          \
	HD DataTypeType & fPost( Direction                 direction ) ;      \
	HD DataTypeType & fPost( Direction::D              direction ) ;      \
	HD DataTypeType & fPost( Direction::DirectionIndex index ) ;          \
	HD DataTypeType & rho() ;                                             \
	HD DataTypeType & rho0() ;                                            \
	HD DataTypeType & rhoBoundary() ;                                     \
	HD DataTypeType & u( Axis axis ) ;                                    \
	HD DataTypeType & uBoundary( Axis axis ) ;                            \
	HD DataTypeType & uT0( Axis axis ) ;                                  \
	HD DataTypeType & u( unsigned axis ) ;                                \
	HD DataTypeType & uBoundary( unsigned axis ) ;                        \
	HD DataTypeType & uT0( unsigned axis ) ;															\
																																				\
	HD unsigned getNodeInTileX() const { return x_ ; }                    \
	HD unsigned getNodeInTileY() const { return y_ ; }                    \
	HD unsigned getNodeInTileZ() const { return z_ ; }                    \



template <class Tile>
class NodeFromTile <Tile, DataStorageMethod::REFERENCE>
{
	public:

		NODE_INTERFACE 

		HD NodeFromTile (Tile & tile, unsigned x, unsigned y=0, unsigned z=0) ;


	private:

		// FIXME: NodeInTileIterator
		// FIXME: different number of coordinates for different lattices
		unsigned x_ ; 
		unsigned y_ ; 
		unsigned z_ ;

		Tile & tile_ ;
} ;



template <class Tile>
class NodeFromTile <Tile, DataStorageMethod::POINTERS>
{
	public:

		NODE_INTERFACE

		HD unsigned getTileIndex() const ;

		HD NodeFromTile
		(
			unsigned x, unsigned y, unsigned z, unsigned tileIndex,
			NodeType            * tiledNodeTypes,
			SolidNeighborMask   * tiledSolidNeighborMasks,
			PackedNodeNormalSet * tiledNodeNormals,
			DataTypeType        * tiledAllValues
		) ;


	private:
		
		HD unsigned computeNodeIndex() const ;
		HD unsigned computeNodeDataIndex (typename Tile::Data data, 
																			Axis axis = Axis::X) ;
		HD unsigned computeNodeDataIndex (typename Tile::Data data, 
																			Direction::DirectionIndex fIndex) ;

		unsigned x_ ;
		unsigned y_ ;
		unsigned z_ ;

		unsigned tileIndex_ ;

		NodeType            * tiledNodeTypes_ ;
		SolidNeighborMask   * tiledSolidNeighborMasks_ ;
		PackedNodeNormalSet * tiledNodeNormals_ ;
		DataTypeType        * tiledAllValues_ ;
} ;



template <class Tile>
class NodeFromTile <Tile, DataStorageMethod::COPY>
: public NodeFromTile <Tile, DataStorageMethod::POINTERS>
{
	public:

		HD NodeFromTile
		(
			unsigned x, unsigned y, unsigned z, unsigned tileIndex,
			NodeType            * tiledNodeTypes,
			SolidNeighborMask   * tiledSolidNeighborMasks,
			PackedNodeNormalSet * tiledNodeNormals,
			typename NodeFromTile <Tile, DataStorageMethod::POINTERS>::DataTypeType
													* tiledAllValues
		) ;

		// Used to connect this object with array in shared memory.
		// TODO: Find some more elegant way.
		DEVICE void registerSharedU 
		(
			typename Tile::DataTypeType (& uGPU) 
												[Tile::LatticeArrangementType::getD()][4][4][4]
		) ;

		HD NodeType & nodeType() ;

		HD typename Tile::DataTypeType & u (unsigned axis) ;
		HD typename Tile::DataTypeType & u (Axis axis) ;
		HD typename Tile::DataTypeType & rho() ;

		HD typename Tile::DataTypeType & f (Direction::DirectionIndex index) ;
		HD typename Tile::DataTypeType & f (Direction direction) ;
		HD typename Tile::DataTypeType & f (Direction::D direction) ;


	private:
	
		typename Tile::DataTypeType rho_ ;     
		typename Tile::DataTypeType u_ [Tile::LatticeArrangementType::getD()] ;

		// Used as reference to array in kernel shared memory.
		typename Tile::DataTypeType (* uGPU_) 
			[Tile::LatticeArrangementType::getD()][4][4][4] ;

		typename Tile::DataTypeType f_ [Tile::LatticeArrangementType::getQ()] ;

		NodeType nodeType_ ;
} ;



#undef NODE_INTERFACE



}



#include "NodeFromTile.hh"



#endif
