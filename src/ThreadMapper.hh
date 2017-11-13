#ifndef THREAD_MAPPER_HH
#define THREAD_MAPPER_HH



namespace microflow
{



template< class TiledLattice >
ThreadMapper< TiledLattice, SingleBlockPerTile, SingleThreadPerNode >::
ThreadMapper( const TiledLattice & tiledLattice )
{
	numberOfTiles_ = tiledLattice.getNOfTiles() ;
}



template< class TiledLattice >
dim3 ThreadMapper< TiledLattice, SingleBlockPerTile, SingleThreadPerNode >::
computeGridDimension() const
{
	return dim3( numberOfTiles_, 1, 1 ) ; // TODO: multidimensional grid for LARGE lattices.
}



template <class TiledLattice>
unsigned ThreadMapper <TiledLattice, SingleBlockPerTile, SingleThreadPerNode>::
computeNumberOfBlocks() const
{
	dim3 gridDim = computeGridDimension() ;

	return gridDim.x * gridDim.y * gridDim.z ;
}



template <class TiledLattice>
unsigned ThreadMapper <TiledLattice, SingleBlockPerTile, SingleThreadPerNode>::
computeNumberOfWarps() const
{
	return computeNumberOfWarpsPerBlock() * computeNumberOfBlocks() ;
}



template <class TiledLattice>
unsigned ThreadMapper <TiledLattice, SingleBlockPerTile, SingleThreadPerNode>::
computeNumberOfWarpsPerBlock() const
{
	dim3 blockDim = computeBlockDimension() ;

	return (blockDim.x * blockDim.y * blockDim.z) / 32 ;
}



template< class TiledLattice > 
INLINE HD
constexpr unsigned ThreadMapper< TiledLattice, SingleBlockPerTile, SingleThreadPerNode >::
getBlockDimX()
{
	return
	(
		3 == TiledLattice::LatticeArrangementType::getD() || 
		2 == TiledLattice::LatticeArrangementType::getD()  
			? TiledLattice::TileType::getNNodesPerEdge() : 0u-1u
	) ;
}



template< class TiledLattice > 
INLINE HD
constexpr unsigned ThreadMapper< TiledLattice, SingleBlockPerTile, SingleThreadPerNode >::
getBlockDimY()
{
	return
	(
		3 == TiledLattice::LatticeArrangementType::getD() || 
		2 == TiledLattice::LatticeArrangementType::getD()  
			? TiledLattice::TileType::getNNodesPerEdge() : 0u-1u
	) ;
}



template< class TiledLattice > 
INLINE HD
constexpr unsigned ThreadMapper< TiledLattice, SingleBlockPerTile, SingleThreadPerNode >::
getBlockDimZ()
{
	return
	(
		3 == TiledLattice::LatticeArrangementType::getD() 
			? TiledLattice::TileType::getNNodesPerEdge() :
		(2 == TiledLattice::LatticeArrangementType::getD()  ?  1 : 0u-1u
		)
	) ;
}



template< class TiledLattice > 
dim3 ThreadMapper< TiledLattice, SingleBlockPerTile, SingleThreadPerNode >::
computeBlockDimension() const
{
	typedef typename TiledLattice::LatticeArrangementType LatticeArrangementType ;

	dim3 result (0u-1u, 0u-1u, 0u-1u) ;

	if ( 
			3 == LatticeArrangementType::getD() ||
			2 == LatticeArrangementType::getD()
		 )
	{
		result = dim3 (getBlockDimX(), getBlockDimY(), getBlockDimZ()) ;
	}
	else
	{
		THROW ("Unsupported number of dimensions") ;
	}

	return result ;
}



}
#endif
