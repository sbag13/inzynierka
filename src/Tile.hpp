#ifndef TILE_HPP
#define TILE_HPP



#include <cstddef>


#include "TileTraitsCommon.hpp"
#include "Axis.hpp"
#include "Direction.hpp"
#include "TileLayout.hpp"
#include "NodeCalculator.hpp"
#include "PackedNodeNormalSet.hpp"
#include "SolidNeighborMask.hpp"
#include "NodeFromTile.hpp"
#include "cudaPrefix.hpp"
#include "TileDataArrangement.hpp"



namespace microflow
{



template< class LatticeArrangement, class DataType,
					unsigned Edge,
					template<class T> class Storage,
					TileDataArrangement DataArrangement >
class TileBase : public TileTraitsCommon< Edge, LatticeArrangement::getD() >
{
	public:
		typedef TileTraitsCommon< Edge, LatticeArrangement::getD() >  TraitsType ;
		typedef LatticeArrangement LatticeArrangementType ;
		typedef DataType DataTypeType ;

		static constexpr TileDataArrangement tileDataArrangement = DataArrangement ;

		// Numbers of probability distrubution function values f_i (PDFs)
		HD static constexpr unsigned getNFsPerTile() ;
		HD static constexpr unsigned getNFsPerNode() ;

		// Number of all values (PDFs, all velocities and densities)
		HD static constexpr unsigned getNValuesPerNode() ;
		HD static constexpr unsigned getNValuesPerTile() ;

		enum class Data
		{
			RHO = 0,
			RHO_T0,
			RHO_BOUNDARY,
			U,
			U_T0,
			U_BOUNDARY,
			F,
			F_POST,
		} ;

		// TODO: Find better names for the below methods...

		//FIXME: There are no checks, whether one calls wrong combination of parameters, i.e.
		// computeTileDataBeginIndex (Data::F_POST, Axis::Z)
		HD static constexpr unsigned 
		computeTileDataBeginIndex (unsigned tileIndex) ;

		HD static constexpr unsigned 
		computeTileDataBeginIndex (unsigned tileIndex, 
															 Data data,
															 Axis uAxis = Axis::X ) ;

		HD static constexpr unsigned 
		computeTileDataBeginIndex (unsigned tileIndex, 
															 Data data,
															 Direction::DirectionIndex fIndex ) ;

		// The below methods compute offsets from computeTileDataBeginIndex (unsigned tileIndex).
		HD static constexpr unsigned
		computeDataBlockInTileIndex (Data data, 
																 Axis uAxis, Direction::DirectionIndex fIndex) ;

		HD static constexpr unsigned
		computeDataBlockInTileIndex (Data data, Axis uAxis = Axis::X) ;

		HD static constexpr unsigned
		computeDataBlockInTileIndex (Data data, Direction::DirectionIndex fIndex) ;


		// The below methods compute offsets from the beginnig of allValues_ array.
		HD static constexpr unsigned
		computeNodeDataIndex (unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
													unsigned tileIndex,
													Data data, Axis axis = Axis::X) ;

		HD static constexpr unsigned
		computeNodeDataIndex (unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
													unsigned tileIndex,
													Data data, Direction::DirectionIndex fIndex) ;

		// Computes offset inside linear array containing Edge*Edge*Edge values.
		// The arrangement of values is defined by TileDataArrangement template parameter.
		HD static constexpr unsigned
		computeIndexInFArray (unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
													Direction::DirectionIndex fIndex) ;



		HD TileBase( TileIterator tileIndex,
								 Storage< NodeType > & nodeTypes,
								 Storage< PackedNodeNormalSet > & nodeNormals,
								 Storage< SolidNeighborMask > & solidNeighborMasks,
								 Storage< DataType > & allValues,
								 const TileLayout<Storage> & tileLayout
							 ) ;

		HD TileIterator getCurrentTileIndex() const ;

		HD_WARNING_DISABLE 
		HD NodeType * getNodeTypesPtr() ;
		HD_WARNING_DISABLE 
		HD PackedNodeNormalSet * getNodeNormalsPtr() ;
		HD_WARNING_DISABLE 
		HD SolidNeighborMask * getNodeSolidNeighborMasksPtr() ;

		HD DataType * getFPtr( Direction direction ) ;
		HD DataType * getFPostPtr( Direction direction ) ;
		HD DataType * getRhoPtr() ;
		HD DataType * getRho0Ptr() ;
		HD DataType * getRhoBoundaryPtr() ;
		HD DataType * getUPtr( Axis axis ) ;
		HD DataType * getUT0Ptr( Axis axis ) ;
		HD DataType * getUBoundaryPtr( Axis axis ) ;

		
		typedef NodeType NodeTypeArray [Edge][Edge][Edge] ;
		typedef DataType DataTypeArray [Edge][Edge][Edge] ;
		typedef DataType VelocityArray [LatticeArrangement::getD()][Edge][Edge][Edge] ;
		typedef DataType FArray        [LatticeArrangement::getQ()][Edge][Edge][Edge] ;

		typedef PackedNodeNormalSet NodeNormalsArray [Edge][Edge][Edge] ;
		typedef SolidNeighborMask NodeSolidNeighborMasksArray [Edge][Edge][Edge] ;

		HD NodeTypeArray & getNodeTypes() ;
		HD NodeNormalsArray & getNodeNormals() ;
		HD NodeSolidNeighborMasksArray & getNodeSolidNeighborMasks() ;

		HD DataTypeArray & f( Direction direction ) ;
		HD DataTypeArray & fPost( Direction direction ) ;
		HD DataTypeArray & rho() ;
		HD DataTypeArray & rho0() ;
		HD DataTypeArray & rhoBoundary() ;
		HD DataTypeArray & u( Axis axis ) ;
		HD DataTypeArray & uT0( Axis axis ) ;
		HD DataTypeArray & uBoundary( Axis axis ) ;

		HD VelocityArray & u() ;
		HD VelocityArray & uT0() ;
		HD VelocityArray & uBoundary() ;

		HD FArray & f() ;
		HD FArray & fPost() ;

		// This method allows to access values for single node.
		// Should be very efficient, when DataStorageMethod::REFERENCE is used.
		/*
			WARNING !!!
			NodeFromTile internally uses a reference to Tile, thus it is not possible
			to use it WITHOUT Tile object. Use of NodeFromTile without Tile results in
			random segmentation faults !!!
		*/
		template <DataStorageMethod DataStorage = DataStorageMethod::REFERENCE>
		HD 
		NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
									DataStorage>
		getNode( unsigned x, unsigned y, unsigned z ) ;

		typedef NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
									DataStorageMethod::REFERENCE> DefaultNodeType ;

		HD TileBase getNeighbor( Direction direction ) ;
		HD bool isEmpty() const ;


	protected:

		// We assume, that all computations are done on some kind of vector machines
		// (SSE/AVX/CUDA), thus the same kind of values (ie. f_i function) for 
		// neighbor nodes are stored in neighbor memory locations.

		//TODO: maybe instead of raw arrays there should be simple reference of TiledLattice ?
		Storage< NodeType > & nodeTypes_ ;
		Storage< PackedNodeNormalSet > & nodeNormals_ ;
		Storage< SolidNeighborMask > & solidNeighborMasks_ ;
		Storage< DataType > & allValues_ ;
		TileIterator currentTileIndex_ ;

		const TileLayout<Storage> & tileLayout_ ;

		HD unsigned computeTileDataBeginIndex() const ;
		HD_WARNING_DISABLE
		HD DataType * getPtr( unsigned blockIndex ) ;

		template< class TargetType = DataTypeArray >
		HD TargetType & castPtr( DataType * ptr ) ;


	private:

		// Hack to allow getNode() specialization.
		template <DataStorageMethod DataStorage> class StorageWrapper {} ;

		HD
		NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
								 DataStorageMethod::REFERENCE>
		getNodeHelper (unsigned x, unsigned y, unsigned z, 
									 StorageWrapper<DataStorageMethod::REFERENCE>) ;

		HD
		NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
								 DataStorageMethod::POINTERS>
		getNodeHelper (unsigned x, unsigned y, unsigned z, 
									 StorageWrapper<DataStorageMethod::POINTERS>) ;


				
} ;



#define TILE_METHODS( storage, dataArrangement )                            \
private: \
typedef TileBase< LatticeArrangement, DataType, Edge, storage, dataArrangement > BaseType ;  \
using   BaseType::tileLayout_ ; \
public: \
using BaseType::tileDataArrangement ;                                       \
using BaseType::getNFsPerTile ;                                             \
using BaseType::getNFsPerNode ;                                             \
using BaseType::getNValuesPerNode ;                                         \
using BaseType::getNValuesPerTile ;                                         \
using BaseType::getCurrentTileIndex ;                                       \
using BaseType::getNodeTypesPtr ;                                           \
using BaseType::getNodeNormalsPtr ;                                         \
using BaseType::getFPtr           ;                                         \
using BaseType::getFPostPtr       ;                                         \
using BaseType::getRhoPtr         ;                                         \
using BaseType::getRho0Ptr        ;                                         \
using BaseType::getRhoBoundaryPtr ;                                         \
using BaseType::getUPtr           ;                                         \
using BaseType::getUT0Ptr         ;                                         \
using BaseType::getUBoundaryPtr   ;                                         \
using BaseType::getNodeTypes ;                                              \
using BaseType::getNodeNormals ;                                            \
using BaseType::f           ;                                               \
using BaseType::fPost       ;                                               \
using BaseType::rho         ;                                               \
using BaseType::rho0        ;                                               \
using BaseType::rhoBoundary ;                                               \
using BaseType::u           ;                                               \
using BaseType::uT0         ;                                               \
using BaseType::uBoundary   ;                                               \
using BaseType::getNode ;                                                   



template <class LatticeArrangement, class DataType,
					unsigned Edge,
					template<class T> class Storage ,
					TileDataArrangement DataArrangement>
class Tile : public TileBase< LatticeArrangement, DataType, Edge, Storage, DataArrangement >
{
	public:
	

		TILE_METHODS (Storage, DataArrangement)


		typedef typename BaseType::TraitsType TraitsType ;
		// TODO: unused ? Used only in tests as for now...
		typedef LatticeArrangement LatticeArrangementType ;
		typedef DataType           DataTypeType ;

		HD Tile( TileIterator tileIndex,
					const TileLayout<Storage> & tileLayout,
					Storage< NodeType > & nodeTypes,
					Storage< PackedNodeNormalSet > & nodeNormals,
					Storage< SolidNeighborMask > & solidNeighborMasks,
					Storage< DataType > & allValues
				) ;

		// Coordinates of tile corner (the node with minimal coordinates)
		Coordinates getCornerPosition() const ;
} ;



template< class LatticeArrangement, class DataType,
					unsigned Edge,
					TileDataArrangement DataArrangement
				>
class Tile <LatticeArrangement, DataType, Edge, StorageInKernel, DataArrangement> 
: public TileBase <LatticeArrangement, DataType, Edge, StorageInKernel, DataArrangement>
{
	public:


		TILE_METHODS (StorageInKernel, DataArrangement)


		HD Tile( TileIterator tileIndex,
						 StorageInKernel< NodeType > & nodeTypes,
						 StorageInKernel< PackedNodeNormalSet > & nodeNormals,
						 StorageInKernel< SolidNeighborMask > & solidNeighborMasks,
						 StorageInKernel< DataType > & allValues,
						 const TileLayout< StorageInKernel > & tileLayout
				) ;
} ;



#undef TILE_METHODS



}



#include "Tile.hh"



#endif
