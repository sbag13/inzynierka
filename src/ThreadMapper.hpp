#ifndef THREAD_MAPPER_HPP
#define THREAD_MAPPER_HPP



namespace microflow
{



class SingleBlockPerTile
{} ;



class SingleThreadPerNode
{} ;



template< class TiledLattice,
					class BlockMapping,
					class ThreadMapping >
class ThreadMapper
{
} ;



template< class TiledLattice >
class ThreadMapper< TiledLattice, SingleBlockPerTile, SingleThreadPerNode >
{
	public:
		ThreadMapper( const TiledLattice & tiledLattice ) ;

		dim3 computeGridDimension() const ;
		dim3 computeBlockDimension() const ;

		unsigned computeNumberOfBlocks() const ;
		unsigned computeNumberOfWarps() const ;
		unsigned computeNumberOfWarpsPerBlock() const ;

		//TODO: Poor mans fix, because dim3 is not literal and can not be constexpr.
		static HD constexpr unsigned getBlockDimX() ;
		static HD constexpr unsigned getBlockDimY() ;
		static HD constexpr unsigned getBlockDimZ() ;

	private:
		unsigned numberOfTiles_ ;
} ;



}



#include "ThreadMapper.hh"



#endif
