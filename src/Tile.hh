#ifndef TILE_HH
#define TILE_HH



#include <limits>



#include "LatticeArrangement.hpp"
#include "MultidimensionalMappers.hpp"



namespace microflow
{



#define TEMPLATE_TILE_NO_INLINE                      \
template <class LatticeArrangement, class DataType,  \
					unsigned Edge,                             \
				 	template<class T> class Storage,					 \
					TileDataArrangement DataArrangement>



#define TEMPLATE_TILE \
TEMPLATE_TILE_NO_INLINE   INLINE



#define TILE   \
Tile <LatticeArrangement, DataType, Edge, Storage, DataArrangement>



#define TILE_BASE   \
TileBase <LatticeArrangement, DataType, Edge, Storage, DataArrangement>



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
getNFsPerNode()
{
	// Two copies of f_i functions: f and fPost */
	return 2 * LatticeArrangement::getQ() ; 
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
getNFsPerTile()
{
	return getNFsPerNode() * TraitsType::getNNodesPerTile() ;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
getNValuesPerNode()
{
	/*
		In allValues array three sets of variables are stored after f_i and f_i post:
			- velocity and density
			- t0 velocity and density
			- boundary velocity and density
		Each set occupies 3 or 4 contiguous blocks depending on number of dimensions.
	*/
	return getNFsPerNode() +  \
				 3 * (LatticeArrangement::getD() /* velocity coordinates */ + 1 /* rho */) ;	
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
getNValuesPerTile()
{
	return TraitsType::getNNodesPerTile() * getNValuesPerNode() ;
}



TEMPLATE_TILE
bool TILE_BASE::
isEmpty() const
{
	return EMPTY_TILE == getCurrentTileIndex() ;
}



TEMPLATE_TILE
TILE_BASE  TILE_BASE::
getNeighbor( Direction direction )
{
	auto neighborIndex = tileLayout_.getNeighborIndex( getCurrentTileIndex() , direction ) ;

	return TileBase( neighborIndex, 
									 nodeTypes_, 
									 nodeNormals_, solidNeighborMasks_,
									 allValues_, tileLayout_ ) ;
}



TEMPLATE_TILE
Coordinates TILE::
getCornerPosition() const
{
	typename TileLayout<Storage>::NonEmptyTile tile = 
				tileLayout_.getTile( getCurrentTileIndex() ) ;
	return tile.getCornerPosition() ;
}



TEMPLATE_TILE
TILE::
Tile( size_t tileIndex,
			const TileLayout<Storage> & tileLayout ,
			Storage< NodeType > & nodeTypes,
			Storage< PackedNodeNormalSet > & nodeNormals,
			Storage< SolidNeighborMask > & solidNeighborMasks,
			Storage< DataType > & allValues 
		) :
	TILE_BASE::TileBase( tileIndex, 
											 nodeTypes, 
											 nodeNormals, solidNeighborMasks,
											 allValues, tileLayout )
{
}



template <class LatticeArrangement, class DataType, unsigned Edge,
					TileDataArrangement DataArrangement>
INLINE
Tile <LatticeArrangement, DataType, Edge, StorageInKernel, DataArrangement>::
Tile( size_t tileIndex,
			StorageInKernel< NodeType > & nodeTypes,
			StorageInKernel< PackedNodeNormalSet > & nodeNormals,
			StorageInKernel< SolidNeighborMask > & solidNeighborMasks,
			StorageInKernel< DataType > & allValues,
			const TileLayout< StorageInKernel > & tileLayout
		) :
	TileBase <LatticeArrangement, DataType, Edge, StorageInKernel, DataArrangement>::
			TileBase( tileIndex, 
								nodeTypes, 
								nodeNormals, solidNeighborMasks,
								allValues, tileLayout )
{
}



TEMPLATE_TILE
TILE_BASE::
TileBase( 
		size_t tileIndex,
		Storage< NodeType > & nodeTypes,
		Storage< PackedNodeNormalSet > & nodeNormals,
		Storage< SolidNeighborMask > & solidNeighborMasks,
		Storage< DataType > & allValues,
		const TileLayout<Storage> & tileLayout ) :
	nodeTypes_ ( nodeTypes ),
	nodeNormals_ ( nodeNormals ),
	solidNeighborMasks_ ( solidNeighborMasks ),
	allValues_ ( allValues ),
	currentTileIndex_( tileIndex ),
	tileLayout_( tileLayout )
{
}



TEMPLATE_TILE
TileIterator TILE_BASE::
getCurrentTileIndex() const
{
	return currentTileIndex_ ;
}



/*
	All f_i functions are stored one after the other 
	at the beginning of tile data.
*/
TEMPLATE_TILE
DataType * TILE_BASE::
getFPtr( Direction direction )
{
		return getPtr (computeDataBlockInTileIndex (Data::F, 
																LatticeArrangement::getIndex (direction.get()) ) ) ;
}



/*
	f_i post values are stored after all f_i.
*/
TEMPLATE_TILE
DataType * TILE_BASE::
getFPostPtr( Direction direction )
{
		return getPtr (computeDataBlockInTileIndex (Data::F_POST, 
																LatticeArrangement::getIndex (direction.get()) ) ) ;
}



TEMPLATE_TILE
NodeType * TILE_BASE::
getNodeTypesPtr()
{
		const unsigned tileBegin = TraitsType::computeTileNodesBeginIndex (getCurrentTileIndex()) ;
		return & (nodeTypes_[tileBegin] ) ;
}



TEMPLATE_TILE
PackedNodeNormalSet * TILE_BASE::
getNodeNormalsPtr()
{
		const unsigned tileBegin = TraitsType::computeTileNodesBeginIndex (getCurrentTileIndex()) ;
		return & (nodeNormals_[tileBegin] ) ;
}



TEMPLATE_TILE
SolidNeighborMask * TILE_BASE::
getNodeSolidNeighborMasksPtr()
{
		const unsigned tileBegin = TraitsType::computeTileNodesBeginIndex (getCurrentTileIndex()) ;
		return & (solidNeighborMasks_[tileBegin] ) ;
}



TEMPLATE_TILE
DataType * TILE_BASE::
getRhoPtr()
{
	return getPtr (computeDataBlockInTileIndex (Data::RHO)) ;
}



TEMPLATE_TILE
DataType * TILE_BASE::
getRho0Ptr()
{
	return getPtr (computeDataBlockInTileIndex (Data::RHO_T0)) ;
}



TEMPLATE_TILE
DataType * TILE_BASE::
getRhoBoundaryPtr()
{
	return getPtr (computeDataBlockInTileIndex (Data::RHO_BOUNDARY)) ;
}



TEMPLATE_TILE
DataType * TILE_BASE::
getUPtr( Axis axis )
{
	return getPtr (computeDataBlockInTileIndex (Data::U, axis)) ;
}



TEMPLATE_TILE
DataType * TILE_BASE::
getUT0Ptr( Axis axis )
{
	return getPtr (computeDataBlockInTileIndex (Data::U_T0, axis)) ;
}



TEMPLATE_TILE
DataType * TILE_BASE::
getUBoundaryPtr( Axis axis )
{
	return getPtr (computeDataBlockInTileIndex (Data::U_BOUNDARY, axis)) ;
}



TEMPLATE_TILE
typename TILE_BASE::NodeTypeArray & 
TILE_BASE::
getNodeTypes()
{
	NodeType * nodeTypesPtr = getNodeTypesPtr() ;

	return *reinterpret_cast< typename TILE_BASE::NodeTypeArray * >(nodeTypesPtr) ;
}



TEMPLATE_TILE
typename TILE_BASE::NodeNormalsArray & 
TILE_BASE::
getNodeNormals()
{
	PackedNodeNormalSet * nodeNormalsPtr = getNodeNormalsPtr() ;

	return *reinterpret_cast< typename TILE_BASE::NodeNormalsArray * >(nodeNormalsPtr) ;
}



TEMPLATE_TILE
typename TILE_BASE::NodeSolidNeighborMasksArray & 
TILE_BASE::
getNodeSolidNeighborMasks()
{
	SolidNeighborMask * solideNeighborMasksPtr = getNodeSolidNeighborMasksPtr() ;

	return *reinterpret_cast< typename TILE_BASE::NodeSolidNeighborMasksArray * >
							( solideNeighborMasksPtr ) ;
}



TEMPLATE_TILE
typename TILE_BASE::DataTypeArray &  TILE_BASE::
f( Direction direction )
{
	DataType * fPtr = getFPtr( direction ) ;

	return castPtr( fPtr ) ;
}



TEMPLATE_TILE
typename TILE_BASE::DataTypeArray &  TILE_BASE::
fPost( Direction direction )
{
	DataType * fPostPtr = getFPostPtr( direction ) ;

	return castPtr( fPostPtr ) ;
}



TEMPLATE_TILE
typename TILE_BASE::DataTypeArray &  TILE_BASE::
rho()
{
	return castPtr( getRhoPtr() ) ;
}



TEMPLATE_TILE
typename TILE_BASE::DataTypeArray &  TILE_BASE::
rho0()
{
	return castPtr( getRho0Ptr() ) ;
}



TEMPLATE_TILE
typename TILE_BASE::DataTypeArray &  TILE_BASE::
rhoBoundary()
{
	return castPtr( getRhoBoundaryPtr() ) ;
}



TEMPLATE_TILE
typename TILE_BASE::DataTypeArray &  TILE_BASE::
u( Axis axis )
{
	return castPtr( getUPtr(axis) ) ;
}



TEMPLATE_TILE
typename TILE_BASE::DataTypeArray &  TILE_BASE::
uT0( Axis axis )
{
	return castPtr( getUT0Ptr(axis) ) ;
}



TEMPLATE_TILE
typename TILE_BASE::DataTypeArray &  TILE_BASE::
uBoundary( Axis axis )
{
	return castPtr( getUBoundaryPtr(axis) ) ;
}



TEMPLATE_TILE
typename TILE_BASE::VelocityArray &  TILE_BASE::
u()
{
	return castPtr<	VelocityArray >( getUPtr(Axis::X) ) ;
}



TEMPLATE_TILE
typename TILE_BASE::VelocityArray &  TILE_BASE::
uT0()
{
	return castPtr<	VelocityArray >( getUT0Ptr(Axis::X) ) ;
}



TEMPLATE_TILE
typename TILE_BASE::VelocityArray &  TILE_BASE::
uBoundary()
{
	return castPtr<	VelocityArray >( getUBoundaryPtr(Axis::X) ) ;
}



TEMPLATE_TILE
typename TILE_BASE::FArray &  TILE_BASE::
f()
{
	constexpr Direction::D c0 = LatticeArrangement::c[0] ;
	return castPtr<	FArray >( getFPtr( c0 ) ) ;

}



TEMPLATE_TILE
typename TILE_BASE::FArray &  TILE_BASE::
fPost()
{
	constexpr Direction::D c0 = LatticeArrangement::c[0] ;

	return castPtr<	FArray >( getFPostPtr( c0 ) ) ;
}



TEMPLATE_TILE_NO_INLINE 
template <DataStorageMethod DataStorage>
INLINE							
NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
							DataStorage>
TILE_BASE::
getNode( unsigned x, unsigned y, unsigned z )
{
	return getNodeHelper (x,y,z, StorageWrapper<DataStorage>()) ;
}



TEMPLATE_TILE
NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
							DataStorageMethod::REFERENCE>
TILE_BASE::
getNodeHelper( unsigned x, unsigned y, unsigned z, StorageWrapper<DataStorageMethod::REFERENCE> )
{
	return 
		NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
									DataStorageMethod::REFERENCE> (*this, x,y,z) ;
}



TEMPLATE_TILE
NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
							DataStorageMethod::POINTERS>
TILE_BASE::
getNodeHelper( unsigned x, unsigned y, unsigned z, StorageWrapper<DataStorageMethod::POINTERS> )
{
	return 
		NodeFromTile <TileBase<LatticeArrangement,DataType,Edge,Storage,DataArrangement>, 
									DataStorageMethod::POINTERS> 
									(
										x,y,z, 
										currentTileIndex_,
										& (nodeTypes_[0]),
										& (solidNeighborMasks_[0]),
										& (nodeNormals_[0]),
										& (allValues_[0])
									) ;
}



TEMPLATE_TILE
unsigned TILE_BASE::
computeTileDataBeginIndex() const
{
	return computeTileDataBeginIndex (getCurrentTileIndex()) ;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeTileDataBeginIndex (unsigned tileIndex)
{
	return tileIndex * getNValuesPerTile() ;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeTileDataBeginIndex (unsigned tileIndex, Data data, Axis axis)
{
	return computeTileDataBeginIndex (tileIndex) +
				 computeDataBlockInTileIndex (data, axis, Direction::DirectionIndex(0)) ;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeTileDataBeginIndex (unsigned tileIndex, Data data, 
													 Direction::DirectionIndex fIndex)
{
	return computeTileDataBeginIndex (tileIndex) +
				 computeDataBlockInTileIndex (data, Axis::X, fIndex) ;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeDataBlockInTileIndex (Data data, Axis axis)
{
	return computeDataBlockInTileIndex (data, axis, Direction::DirectionIndex(0) ) ;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeDataBlockInTileIndex (Data data, Direction::DirectionIndex fIndex)
{
	return computeDataBlockInTileIndex (data, Axis(0), fIndex ) ;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeDataBlockInTileIndex (Data data, Axis axis, Direction::DirectionIndex fIndex)
{
	return
			TraitsType::getNNodesPerTile() * (

			(Data::F            == data) ? 0 * LatticeArrangement::getQ() + fIndex  : (

			(Data::F_POST       == data) ? 1 * LatticeArrangement::getQ() + fIndex  : (
			
			(Data::RHO          == data) ? 2 * LatticeArrangement::getQ() + 0  : (

			(Data::RHO_T0       == data) ? 2 * LatticeArrangement::getQ() + 1  : (

			(Data::RHO_BOUNDARY == data) ? 2 * LatticeArrangement::getQ() + 2  : (

			(Data::U            == data) ? 2 * LatticeArrangement::getQ() + 3 + 
																							 	0 * LatticeArrangement::getD() + 
																									static_cast<unsigned>(axis)     : (

			(Data::U_T0         == data) ? 2 * LatticeArrangement::getQ() + 3 + 
																							 	1 * LatticeArrangement::getD() + 
																									static_cast<unsigned>(axis)     : (

			(Data::U_BOUNDARY   == data) ? 2 * LatticeArrangement::getQ() + 3 + 
																							 	2 * LatticeArrangement::getD() + 
																									static_cast<unsigned>(axis)     : 

			(unsigned)(-1) )))))))
			)
		;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeNodeDataIndex (unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
											unsigned tileIndex,
											Data data, Axis axis)
{
	return computeTileDataBeginIndex   (tileIndex) +
				 computeDataBlockInTileIndex (data, axis) +
				 TraitsType::computeNodeInTileIndex (nodeInTileX, nodeInTileY, nodeInTileZ) ;
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeNodeDataIndex (unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
											unsigned tileIndex,
											Data data, Direction::DirectionIndex fIndex)
{
	return computeTileDataBeginIndex   (tileIndex) +
				 computeDataBlockInTileIndex (data, fIndex) +
				 computeIndexInFArray (nodeInTileX, nodeInTileY, nodeInTileZ, fIndex) ;
}



// Helper class - allows to specialize computeFInTileIndex(...) method.
// TODO: Since there is only one template parameter, maybe we can use only function
//				specialization (without class) ?
template <TileDataArrangement DataArrangement>
class IndexCalculator
{
	public:
		HD static 
		constexpr unsigned
			computeFInTileIndex 
			(
				unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
				unsigned tileEdge,
				Direction::DirectionIndex fIndex
			) ;
} ;



template<>
HD INLINE
constexpr unsigned IndexCalculator <TileDataArrangement::XYZ>::
computeFInTileIndex
(
	unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
	unsigned tileEdge, Direction::DirectionIndex fIndex __attribute__((unused))
)
{
	return XYZ::linearize (nodeInTileX, nodeInTileY, nodeInTileZ,
												 tileEdge, tileEdge, tileEdge ) ;
}



template<>
HD INLINE
constexpr unsigned IndexCalculator <TileDataArrangement::OPT_1>::
computeFInTileIndex
(
	unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
	unsigned tileEdge, Direction::DirectionIndex fIndex
)
{
#define linearizeXYZ XYZ::linearize (nodeInTileX, nodeInTileY, nodeInTileZ, tileEdge, tileEdge, tileEdge )
#define linearizeYXZ YXZ::linearize (nodeInTileX, nodeInTileY, nodeInTileZ, tileEdge, tileEdge, tileEdge )
#define linearizeZigzagNE ZigzagNE::linearize (nodeInTileX, nodeInTileY, nodeInTileZ, tileEdge, tileEdge, tileEdge )

	// TODO: Use Direction instead of DirectionIndex.

	return (fIndex == 1 || fIndex == 3 || (fIndex >= 9 && fIndex <= 14))  ? linearizeYXZ : 
					( (fIndex == 7 ||  fIndex == 8)  ?  linearizeZigzagNE 	: linearizeXYZ ) ;

#undef linearizeXYZ
#undef linearizeYXZ
#undef linearizeZigzagNE
}



TEMPLATE_TILE
constexpr unsigned TILE_BASE::
computeIndexInFArray (unsigned nodeInTileX, unsigned nodeInTileY, unsigned nodeInTileZ,
											Direction::DirectionIndex fIndex)
{
	return IndexCalculator<DataArrangement>::
				 computeFInTileIndex (nodeInTileX, nodeInTileY, nodeInTileZ, Edge, fIndex) ;
}




/*
	All values for a full tile are stored in consecutive locations.
	Thus, all values of the same type (for example single f_i or rho) occupy
	nNodesPerTile consecutive places in allValues_ array.
*/
TEMPLATE_TILE
DataType * TILE_BASE::
getPtr( unsigned dataIndex )
{
	return & (allValues_ [computeTileDataBeginIndex(getCurrentTileIndex()) + dataIndex ]) ;
}



TEMPLATE_TILE_NO_INLINE
template< class TargetType >
INLINE
TargetType &  TILE_BASE::
castPtr( DataType * ptr )
{
	return *reinterpret_cast<	TargetType * >(ptr) ;	
}



#undef TEMPLATE_TILE
#undef TILE
#undef TILE_BASE


	
}



#endif
