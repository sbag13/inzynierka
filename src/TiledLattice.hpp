#ifndef TILED_LATTICE_HPP
#define TILED_LATTICE_HPP



#include "Tile.hpp"
#include "Storage.hpp"
#include "Settings.hpp"
#include "ExpandedNodeLayout.hpp"
#include "ModificationRhoU.hpp"



//TODO: Interface in below classes needs refinement.



namespace microflow
{



class TiledLatticeBase
{
	public:
		static constexpr unsigned getNNodesPerTileEdge() { return DEFAULT_3D_TILE_EDGE ; } ;
} ;



class TiledLatticeBaseTwoCopies
: public TiledLatticeBase
{
	public:

		TiledLatticeBaseTwoCopies() ;


		enum class ValidCopyID
		{
			NONE, F, FPOST
		} ;

		ValidCopyID getValidCopyID() const ;

		bool isValidCopyID (ValidCopyID validCopyID) const ;
		bool isValidCopyIDNone () const ;
		bool isValidCopyIDF    () const ;
		bool isValidCopyIDFPost() const ;

		void setValidCopyID (ValidCopyID validCopyID) ;

		void setValidCopyIDToNone  () ;
		void setValidCopyIDToF     () ;
		void setValidCopyIDToFPost () ;

		void switchValidCopyID() ;


	private:

		ValidCopyID validCopyID_ ;
} ;



template<class LatticeArrangement, class DataType,
				 template<class T> class Storage,
				 TileDataArrangement DataArrangement>
class TiledLattice ;



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
class TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement>
: public TiledLatticeBaseTwoCopies
{
	public:

		TiledLattice( TileLayout<StorageOnCPU> & tileLayout, 
									ExpandedNodeLayout & expandedNodeLayout,
									const Settings & settings
								) ;

		// TODO: Maybe Tile should be exported from TileLayout ?
		typedef Tile <LatticeArrangement, DataType, TiledLatticeBase::getNNodesPerTileEdge(), 
									StorageOnCPU, DataArrangement> TileType ;
		typedef LatticeArrangement LatticeArrangementType ;
		typedef DataType DataTypeType ;

		// We need the same numbers, as in TileLayout.
		typedef TileIterator Iterator ;
		typedef TileIterator ConstIterator ;

		static constexpr 
		TileDataArrangement LatticeDataArrangement = DataArrangement ;

		unsigned getNOfTiles() const ;

		TileType getTile(Iterator tileIndex) ;
		const TileType getTile(ConstIterator tileIndex) const ;


		Iterator getBeginOfTiles() ;
		Iterator getEndOfTiles() ;
		ConstIterator getBeginOfTiles() const ;
		ConstIterator getEndOfTiles() const ;

		TileLayout<StorageOnCPU> & getTileLayout() ;
		const TileLayout<StorageOnCPU> & getTileLayout() const ;

		bool operator==( const TiledLattice<LatticeArrangement, DataType, StorageOnCPU, 
																				DataArrangement> & tiledLattice ) const ;

		void modify (const ModificationRhoU & modificationRhoU) ;

		// FIXME: Used only for tests of Node with pointers on CPU. Remove when not needed.
		NodeType * getNodeTypesPointer() { return & nodeTypes_[0] ; }
		SolidNeighborMask * getSolidNeighborMasksPointer() { return & solidNeighborMasks_[0] ; }
		PackedNodeNormalSet * getNodeNormalsPointer() { return & nodeNormals_[0] ; }
		DataType * getAllValuesPointer() { return & allValues_[0] ; }


		// Helper methods, may be slow.
		template <class Functor>
		void forEachTile (Functor functor) ;

		template <class Functor>
		void forEachNode (Functor functor) ;


	private:

		//FIXME: I am temporarily removing const specifier from tileLayout_
		//       Reconsider, if const is usefull here.
		TileLayout<StorageOnCPU> & tileLayout_ ;

		const Settings & settings_ ;

		// FIXME: move to BaseType - the same used in all specializations.
		StorageOnCPU< NodeType > nodeTypes_ ;
		// TODO: nodeNormals_ needed only, when there are boundary nodes of type 47-52
		// TODO: inefficient memory usage, nodeNormals are nonzero only for a small 
		//			 number of nodes.
		StorageOnCPU< PackedNodeNormalSet > nodeNormals_ ;
		// TODO: used only for boundary nodes of type 47-52
		StorageOnCPU< SolidNeighborMask > solidNeighborMasks_ ;

		// To minimise the number of pointers passed to functions/kernels,
		// all values are stored in single array. Unfortunately, the functions need to
		// know, how the array is organized. The organization is implemented in Tile class.
		StorageOnCPU< DataType > allValues_ ;

		friend TiledLattice <LatticeArrangement, DataType, StorageOnGPU, DataArrangement> ;
} ;



/*
	Class used to manage GPU memory - copies data between GPU and CPU
*/
template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
class    TiledLattice <LatticeArrangement, DataType, StorageOnGPU, DataArrangement> 
: public TiledLatticeBaseTwoCopies
{
	public:

		typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> 
							TiledLatticeCPU ;
		typedef Tile <LatticeArrangement, DataType, getNNodesPerTileEdge(), StorageOnGPU, 
									DataArrangement> TileType ;
		typedef LatticeArrangement LatticeArrangementType ;

		TiledLattice( TiledLatticeCPU & tiledLatticeCPU, 
									TileLayout< StorageOnGPU > & tileLayout ) ;

		//TODO: need constructor from pointers passed to kernel 
		//      or (seems better now) separate class.

		void copyToCPU( TiledLatticeCPU & tiledLatticeCPU ) const ;
		void copyToCPU() const ;
		void copyFromCPU( const TiledLatticeCPU & tiledLatticeCPU ) ;
		void copyFromCPU() ;

		unsigned getNOfTiles() const ;

		StorageOnGPU< NodeType > & getNodeTypes() ;
		StorageOnGPU< PackedNodeNormalSet > & getNodeNormals() ;
		StorageOnGPU< SolidNeighborMask > & getSolidNeighborMasks() ;
		StorageOnGPU< DataType > & getAllValues() ;
		TileLayout<StorageOnGPU> & getTileLayout() ;

		// Needed for calls of optimized kernels.
		NodeType * getNodeTypesPointer() { return nodeTypes_.getPointer() ; }
		SolidNeighborMask * getSolidNeighborMasksPointer() { return solidNeighborMasks_.getPointer() ; }
		PackedNodeNormalSet * getNodeNormalsPointer() { return nodeNormals_.getPointer() ; }
		DataType * getAllValuesPointer() { return allValues_.getPointer() ; }


	private:
	
		//FIXME: const modifier removed. Is it OK ?
		TileLayout< StorageOnGPU > & tileLayout_ ;

		StorageOnGPU< NodeType > nodeTypes_ ;
		StorageOnGPU< PackedNodeNormalSet > nodeNormals_ ;
		StorageOnGPU< SolidNeighborMask > solidNeighborMasks_ ;
		StorageOnGPU< DataType > allValues_ ;


		TiledLatticeCPU & tiledLatticeCPU_ ; // TODO: maybe unnecessary ?
} ;



/*
	Wrapper for pointers passed to GPU kernel.
*/
template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
class    TiledLattice <LatticeArrangement, DataType, StorageInKernel, DataArrangement> 
: public TiledLatticeBase
{
	public:

		TiledLattice( 
			TiledLattice <LatticeArrangement, DataType, StorageOnGPU, DataArrangement> 
				& tiledLatticeGPU ) ;

		typedef Tile <LatticeArrangement, DataType, TiledLatticeBase::getNNodesPerTileEdge(), 
									StorageInKernel, DataArrangement> TileType ;

		// We need the same numbers, as in TileLayout.
		typedef TileIterator Iterator ;
		typedef TileIterator ConstIterator ;

		HD TileType getTile(Iterator tileIndex) ;


	private:

		// Not reference, because a copy of this class is passed to kernel
		TileLayout<StorageInKernel> tileLayout_ ;

		StorageInKernel< NodeType > nodeTypes_ ;
		StorageInKernel< PackedNodeNormalSet > nodeNormals_ ;
		StorageInKernel< SolidNeighborMask > solidNeighborMasks_ ;
		StorageInKernel< DataType > allValues_ ;
} ;



}



#include "TiledLattice.hh"
#include "TiledLattice.tcc"



#endif
