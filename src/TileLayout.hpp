#ifndef TILE_LAYOUT_HPP
#define TILE_LAYOUT_HPP



#include "NodeLayout.hpp"
#include "Direction.hpp"
#include "TileDefinitions.hpp"
#include "TileIterator.hpp"
#include "TileTraitsCommon.hpp"
#include "Storage.hpp"


#include <vector>
#include <string>
#include <limits>



namespace microflow
{



const NodeType layoutBorderNode = NodeBaseType::MARKER ;

class TilingStatistic ;



//FIXME: Probably ExpandedNodeLayout should be a part of TileLayout, because
//			 TiledLattice requires ExpandedNodeLayout as constructor parameter.
//			 Maybe we should add method expand() to NodeLayout class, which method
//			 creates additional arrays inside NodeLayout ?



template< template<class> class Storage >
class TileLayoutBase
{
	public:

		// This class is only for programmers convenience and should be completely
		// removed during compilation. Maybe I should use only references to 
		// elements from TileLayout.tilesX0_ arrays ?
		class NonEmptyTile : public TileTraitsCommon< DEFAULT_3D_TILE_EDGE, 3u >
		{
			public:
				typedef size_t Iterator ;
				typedef TileIterator TileIndex ;

				HD NonEmptyTile( TileLayoutBase & tileLayoutBase_, TileIndex tileIndex ) ;

				HD Coordinates getCornerPosition() const ;
				HD Coordinates getMapPosition() const ;

				HD Iterator getBeginOfNodes() const ;
				HD Iterator getEndOfNodes() const ;
				HD Coordinates unpack( Iterator iterator ) const ;

				HD TileIndex getIndex() const ;

			private:

				TileLayoutBase & tileLayoutBase_ ;
				TileIndex tileIndex_ ;
		} ;

		TileLayoutBase() {} ; //TODO: remove ?

		template< template<class> class InputStorage >
		TileLayoutBase( 
			InputStorage<size_t> & tilesX0,
			InputStorage<size_t> & tilesY0,
			InputStorage<size_t> & tilesZ0,
			LinearizedMatrix< unsigned int, InputStorage > & tileMap,
			Size & size
		) ;

		HD Size getSize() const ;

		typedef TileIterator Iterator ;
		typedef TileIterator ConstIterator ;

		HD Iterator getBeginOfNonEmptyTiles() ;
		HD Iterator getEndOfNonEmptyTiles() ;
		HD NonEmptyTile getTile( Iterator it ) ;

		HD ConstIterator getBeginOfNonEmptyTiles() const ;
		HD ConstIterator getEndOfNonEmptyTiles() const ;
		HD const NonEmptyTile getTile( Iterator it ) const ;
		// WARNING: slow method, mainly for tests.
		const NonEmptyTile getTile( Coordinates nodeCoordinates ) const ;

		HD ConstIterator getNeighborIndex( Iterator currentTile, Direction direction ) const ;

		HD bool hasNeighbour( const NonEmptyTile & tile, Direction direction ) const ;
		HD NonEmptyTile getNeighbour( const NonEmptyTile & tile, Direction direction ) ;
		HD const NonEmptyTile getNeighbour( const NonEmptyTile & tile, Direction direction ) const ;
	
		HD_WARNING_DISABLE
		HD size_t getNoNonEmptyTiles() const ;

		// Needed for calls of optimized kernels.
		unsigned int * getTileMapPointer() { return tileMap_.getDataPointer() ; }
		size_t * getTilesX0Pointer() { return tilesX0_.getPointer() ; }
		size_t * getTilesY0Pointer() { return tilesY0_.getPointer() ; }
		size_t * getTilesZ0Pointer() { return tilesZ0_.getPointer() ; }

	protected:

		HD_WARNING_DISABLE
		HD Coordinates getNonEmptyTilePosition( typename TileLayoutBase::NonEmptyTile::TileIndex tileIndex ) const ;

		// Unpacked to 3 vectors to simplify transfer to GPU memory as SOA.
		// TODO: build separate object with different memory layout for CPU and GPU ?
		// TODO: maybe shoud be some defined type instead of size_t ?
		Storage< size_t > tilesX0_ ;
		Storage< size_t > tilesY0_ ;
		Storage< size_t > tilesZ0_ ;

		LinearizedMatrix<unsigned int, Storage> tileMap_ ;

		Size size_ ;
} ;



template< template<class T> class Storage >
class TileLayout
{
} ;



template<> class TileLayout< StorageOnGPU > ;



template<>
class TileLayout< StorageOnCPU >
: public TileLayoutBase< StorageOnCPU >
{
	public:

		TileLayout( const NodeLayout & nodeLayout ) ;

		//TODO: remove the two below methods, these duplicate TilingStatistic interface.
		size_t computeNoTilesTotal() const ;
		// TODO: make the function below private.
		bool isTileEmpty(size_t tileX0, size_t tileY0, size_t tileZ0) const ;

		// TODO: I don't like this method here...
		const NodeType & getNodeType( Coordinates coordinates ) const ;

		//TODO: remove saveToVolFile() method and use separate Writer object.
		void saveToVolFile( std::string fileName ) const ;
		NodeLayout generateLayoutWithMarkedTiles() const ;

		TilingStatistic computeTilingStatistic() const ;

		const NodeLayout & getNodeLayout() const ;
		const LinearizedMatrix<unsigned,StorageOnCPU> & getTileMap() const ;

		bool operator==( const TileLayout<StorageOnCPU> & tileLayout ) const ;

	private:

		// TODO: Question - wether a class TileLayout should contain NodeLayout ?
		NodeLayout nodeLayout_ ;

		friend class TileLayout<StorageOnGPU> ;
} ;



template<> class TileLayout< StorageInKernel > ;



/*
	Class used to manage GPU memory - copies data between GPU and CPU
*/
template<>
class TileLayout< StorageOnGPU >
: public TileLayoutBase< StorageOnGPU >
{
	public:
		TileLayout( TileLayout<StorageOnCPU> & tileLayoutCPU ) ;

		void copyToCPU( TileLayout<StorageOnCPU> & tileLayoutCPU ) const ;
		void copyToCPU() const ;
		void copyFromCPU( const TileLayout<StorageOnCPU> & tileLayoutCPU ) ;
		void copyFromCPU() ;

		Size getSize() { return tileLayoutCPU_.getNodeLayout().getSize() ;  }

	private:
		TileLayout< StorageOnCPU > & tileLayoutCPU_ ; // TODO: maybe unnecessary ?

		friend class TileLayout< StorageInKernel > ;
} ;



template<>
class TileLayout< StorageInKernel >
: public TileLayoutBase< StorageInKernel >
{
	private: typedef TileLayoutBase<StorageInKernel> BaseType ;

	public:

		TileLayout( TileLayout<StorageOnGPU> & tileLayoutGPU ) ;
} ;



}



#include "TileLayout.hh"
#include "TileLayout.tcc"



#endif
