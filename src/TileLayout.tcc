#ifndef TILE_LAYOUT_TCC
#define TILE_LAYOUT_TCC



#ifdef __CUDACC__



#include "Logger.hpp"



namespace microflow
{



#define TILE_LAYOUT_TEMPLATE \
inline



#define TILE_LAYOUT_GPU \
TileLayout< StorageOnGPU >



TILE_LAYOUT_TEMPLATE
TILE_LAYOUT_GPU::
TileLayout( TileLayout< StorageOnCPU > & tileLayoutCPU )
: tileLayoutCPU_( tileLayoutCPU )
{
	copyFromCPU( tileLayoutCPU ) ;
}



TILE_LAYOUT_TEMPLATE
void TILE_LAYOUT_GPU::
copyToCPU( TileLayout< StorageOnCPU > & tileLayoutCPU ) const
{
	tileLayoutCPU.tilesX0_ = tilesX0_ ;
	tileLayoutCPU.tilesY0_ = tilesY0_ ;
	tileLayoutCPU.tilesZ0_ = tilesZ0_ ;

	tileLayoutCPU.tileMap_ = tileMap_ ;
	tileLayoutCPU.size_    = size_    ;
}



TILE_LAYOUT_TEMPLATE
void TILE_LAYOUT_GPU::
copyToCPU() const
{
	copyToCPU( tileLayoutCPU_ ) ;
}



TILE_LAYOUT_TEMPLATE
void TILE_LAYOUT_GPU::
copyFromCPU( const TileLayout< StorageOnCPU > & tileLayoutCPU )
{
  tilesX0_ = tileLayoutCPU.tilesX0_ ;
  tilesY0_ = tileLayoutCPU.tilesY0_ ;
  tilesZ0_ = tileLayoutCPU.tilesZ0_ ;

	tileMap_ = tileLayoutCPU.tileMap_ ;
	size_    = tileLayoutCPU.size_    ;
}



TILE_LAYOUT_TEMPLATE
void TILE_LAYOUT_GPU::
copyFromCPU()
{
	copyFromCPU( tileLayoutCPU_ ) ;
}



inline
TileLayout<StorageInKernel>::
TileLayout( TileLayout<StorageOnGPU> & tileLayoutGPU )
: BaseType
( 
	tileLayoutGPU.tilesX0_,
	tileLayoutGPU.tilesY0_,
	tileLayoutGPU.tilesZ0_,
	tileLayoutGPU.tileMap_,
	tileLayoutGPU.size_
)
{
}



}



#undef TILE_LAYOUT_GPU
#undef TILE_LAYOUT_TEMPLATE



#endif // __CUDACC__



#endif
