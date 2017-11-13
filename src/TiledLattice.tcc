#ifndef TILED_LATTICE_TCC
#define TILED_LATTICE_TCC



#ifdef __CUDACC__



namespace microflow
{



#define TILED_LATTICE_TEMPLATE \
template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>  \
inline



#define TILED_LATTICE_GPU \
TiledLattice <LatticeArrangement, DataType, StorageOnGPU, DataArrangement>

#define TILED_LATTICE_KERNEL \
TiledLattice <LatticeArrangement, DataType, StorageInKernel, DataArrangement>



TILED_LATTICE_TEMPLATE
TILED_LATTICE_GPU::
TiledLattice( TiledLatticeCPU & tiledLatticeCPU,
							TileLayout< StorageOnGPU > & tileLayout ) 
:	tileLayout_( tileLayout ),
	tiledLatticeCPU_( tiledLatticeCPU )
{
	copyFromCPU() ;
}



TILED_LATTICE_TEMPLATE
void TILED_LATTICE_GPU::
copyToCPU (TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> 
							& tiledLatticeCPU) const
{
	tiledLatticeCPU.nodeTypes_ = nodeTypes_ ;
	tiledLatticeCPU.allValues_ = allValues_ ;
	tiledLatticeCPU.nodeNormals_ = nodeNormals_ ;
	tiledLatticeCPU.solidNeighborMasks_ = solidNeighborMasks_ ;

	tiledLatticeCPU.setValidCopyID (getValidCopyID()) ;
}



TILED_LATTICE_TEMPLATE
void TILED_LATTICE_GPU::
copyToCPU() const
{
	copyToCPU( tiledLatticeCPU_ ) ;
}



TILED_LATTICE_TEMPLATE
void TILED_LATTICE_GPU::
copyFromCPU (const TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement> 
								& tiledLatticeCPU)
{
	nodeTypes_ = tiledLatticeCPU.nodeTypes_ ;
	allValues_ = tiledLatticeCPU.allValues_ ;
	nodeNormals_ = tiledLatticeCPU.nodeNormals_ ;
	solidNeighborMasks_ = tiledLatticeCPU.solidNeighborMasks_ ;

	setValidCopyID (tiledLatticeCPU.getValidCopyID()) ;
}



TILED_LATTICE_TEMPLATE
void TILED_LATTICE_GPU::
copyFromCPU()
{
	copyFromCPU( tiledLatticeCPU_ ) ;
}



TILED_LATTICE_TEMPLATE
unsigned TILED_LATTICE_GPU::
getNOfTiles() const
{
	return tiledLatticeCPU_.getNOfTiles() ;
}



TILED_LATTICE_TEMPLATE
StorageOnGPU< NodeType > & TILED_LATTICE_GPU::
getNodeTypes()
{
	return nodeTypes_ ;
}



TILED_LATTICE_TEMPLATE
StorageOnGPU< PackedNodeNormalSet > & TILED_LATTICE_GPU::
getNodeNormals()
{
	return nodeNormals_ ;
}



TILED_LATTICE_TEMPLATE
StorageOnGPU< SolidNeighborMask > & TILED_LATTICE_GPU::
getSolidNeighborMasks()
{
	return solidNeighborMasks_ ;
}



TILED_LATTICE_TEMPLATE
StorageOnGPU< DataType > & TILED_LATTICE_GPU::
getAllValues()
{
	return allValues_ ;
}



TILED_LATTICE_TEMPLATE
TileLayout< StorageOnGPU > & TILED_LATTICE_GPU::
getTileLayout()
{
	return tileLayout_ ;
}



TILED_LATTICE_TEMPLATE
TILED_LATTICE_KERNEL::
TiledLattice (TiledLattice <LatticeArrangement, DataType, StorageOnGPU, DataArrangement> 
									& tiledLatticeGPU)
:	tileLayout_( tiledLatticeGPU.getTileLayout() ),
	nodeTypes_( tiledLatticeGPU.getNodeTypes() ),
	nodeNormals_( tiledLatticeGPU.getNodeNormals() ),
	solidNeighborMasks_( tiledLatticeGPU.getSolidNeighborMasks() ),
	allValues_( tiledLatticeGPU.getAllValues() )
{
}



TILED_LATTICE_TEMPLATE
HD TILED_LATTICE_KERNEL::TileType TILED_LATTICE_KERNEL::
getTile( Iterator tileIndex )
{
	return TileType( tileIndex, 
									 nodeTypes_, 
									 nodeNormals_, solidNeighborMasks_, 
									 allValues_, tileLayout_ ) ;
}



#undef TILED_LATTICE_KERNEL
#undef TILED_LATTICE_GPU
#undef TILED_LATTICE_TEMPLATE



}



#endif // __CUDACC__



#endif
