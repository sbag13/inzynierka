#include "TiledLattice.hpp"
#include "LatticeArrangement.hpp"
#include "BoundaryDefinitions.hpp"



namespace microflow
{



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement>
TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement>::
TiledLattice( 
							TileLayout<StorageOnCPU> & tileLayout, 
							ExpandedNodeLayout & expandedNodeLayout,
							const Settings & settings
						) :
tileLayout_( tileLayout ),
settings_ (settings)
{
	const size_t nTiles = tileLayout_.getNoNonEmptyTiles() ;
	const unsigned nValuesPerTile = TileType::getNValuesPerTile() ;

	const size_t nAll = nTiles * nValuesPerTile ;
	const unsigned nNodesPerTile = TileType::getNNodesPerTile() ;
	const unsigned nNodes = nTiles * nNodesPerTile ;

	const size_t 
	requiredMemoryBytes = nAll * sizeof (allValues_[0]) + 
												nNodes * (sizeof (nodeTypes_[0]) + 
																	sizeof (nodeNormals_[0]) +
																	sizeof (solidNeighborMasks_[0])) ;

	logger << "Building TiledLattice, " 
				 << bytesToHuman (requiredMemoryBytes)
				 << " of memory required, physical memory installed "
				 << bytesToHuman (getInstalledPhysicalMemoryInBytes())
				 << " (" << bytesToHuman (getFreePhysicalMemoryInBytes()) << " free)"
				 << ".\n" ;

	allValues_.resize (nAll, 0) ;
	nodeTypes_         .resize (nNodes) ;
	nodeNormals_       .resize (nNodes ) ;
	solidNeighborMasks_.resize (nNodes) ;


	const BoundaryDefinitions & boundaryDefinitions = 
										tileLayout.getNodeLayout().getBoundaryDefinitions() ;

	const Size sizeInNodes = tileLayout_.getNodeLayout().getSize() ;

	for ( auto t = tileLayout.getBeginOfNonEmptyTiles() ;
						 t != tileLayout.getEndOfNonEmptyTiles() ;
						 t ++ )
	{
		auto sourceTile = tileLayout.getTile( t ) ;
		auto targetTile = this->getTile( t ) ;

		//TODO: In nodeTypes_ nodes must be arranged in row order,
		//      but there is no guarantee, that n traverses in this order.
		for (auto n = sourceTile.getBeginOfNodes() ;
							n != sourceTile.getEndOfNodes() ;
							n++ )
		{
			Coordinates globalCoordinates = sourceTile.unpack( n ) ;
			Coordinates tileCoordinates = globalCoordinates - sourceTile.getCornerPosition() ;

			auto targetNode = targetTile.getNode( tileCoordinates.getX(), 
																						tileCoordinates.getY(),
																						tileCoordinates.getZ() ) ;
			
			targetNode.nodeType() = tileLayout.getNodeType( globalCoordinates ) ;

			if (expandedNodeLayout.getSize().areCoordinatesInLimits( globalCoordinates ) )
			{
				targetNode.nodeNormals() = 
							expandedNodeLayout.getNormalVectors( globalCoordinates ) ;

				targetNode.solidNeighborMask() = 
							expandedNodeLayout.getSolidNeighborMask( globalCoordinates ) ;
			}
			else
			{
				targetNode.nodeNormals() = PackedNodeNormalSet() ;
				targetNode.solidNeighborMask() = SolidNeighborMask() ;
			}


			//FIXME: ugly hack for compatibility with Boundary_Set() function by RS.
			// 			 Remember, that in RS code plane with z=0 is used for storing
			//			 readed boundary velocity and density.
			if ( 
					(	(sizeInNodes.getDepth() -1) == globalCoordinates.getZ() )  &&
					targetNode.nodeType().isSolid() 
				 ) 
			{
				continue ;
			}


			DataType pressure = 0 ;
			DataType velocityX = 0 ;
			DataType velocityY = 0 ;
			DataType velocityZ = 0 ;

			if ( targetNode.nodeType().isBoundary() )
			{
				unsigned char boundaryDefinitionIndex = 
													targetNode.nodeType().getBoundaryDefinitionIndex() ;

				pressure = boundaryDefinitions.getBoundaryPressure( boundaryDefinitionIndex ) ;

				velocityX = boundaryDefinitions.getBoundaryVelocityX( boundaryDefinitionIndex ) ;
				velocityY = boundaryDefinitions.getBoundaryVelocityY( boundaryDefinitionIndex ) ;
				velocityZ = boundaryDefinitions.getBoundaryVelocityZ( boundaryDefinitionIndex ) ;
			}

			targetNode.rhoBoundary() = settings.
									transformPressurePhysicalToVolumetricMassDensityLB( pressure ) ;

			targetNode.uBoundary(X) = settings.transformVelocityPhysicalToLB( velocityX ) ;
			targetNode.uBoundary(Y) = settings.transformVelocityPhysicalToLB( velocityY ) ;
			targetNode.uBoundary(Z) = settings.transformVelocityPhysicalToLB( velocityZ ) ;


			//FIXME: whether these values should be set here ? For now required for
			//			 compatibility with clearDataRS().
			targetNode.rho() = 0 ;
			for (unsigned i=0 ; i < LatticeArrangement::getD() ; i++)
			{
				targetNode.u  (i) = 0 ;
				targetNode.uT0(i) = 0 ;
			}
			for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
			{
				targetNode.fPost(q) = 0 ;
				targetNode.f    (q) = 0 ;
			}

		}
	}

	modify (settings.getModificationRhoU()) ;

	setValidCopyIDToF() ;
}



template <class LatticeArrangement, class DataType, TileDataArrangement DataArrangement >
void TiledLattice <LatticeArrangement, DataType, StorageOnCPU, DataArrangement>::
modify (const ModificationRhoU & modificationRhoU)
{
	// Since in Tile::Node class there is a reference to Tile object 
	// then we can not provide method, which returns only Tile::Node.
	#define GET_NODE                                                                 \
		auto tileLayout = getTileLayout() ;                                            \
		TileIterator tileIndex = tileLayout.getTile (m.coordinates).getIndex() ;       \
		auto tile = getTile (tileIndex) ;                                              \
		Coordinates nodeInTileCoordinates = m.coordinates - tile.getCornerPosition() ; \
																																									 \
		auto node = tile.getNode                                                       \
							(                                                                    \
							 nodeInTileCoordinates.getX(),                                       \
							 nodeInTileCoordinates.getY(),                                       \
							 nodeInTileCoordinates.getZ()                                        \
							) ;                                                                  \

	for (auto m : modificationRhoU.uPhysical)
	{
		GET_NODE ;

		node.u (X) = settings_.transformVelocityPhysicalToLB (m.value[0]) ;
		node.u (Y) = settings_.transformVelocityPhysicalToLB (m.value[1]) ;
		node.u (Z) = settings_.transformVelocityPhysicalToLB (m.value[2]) ;
	}

	for (auto m : modificationRhoU.uBoundaryPhysical)
	{
		GET_NODE ;

		node.uBoundary (X) = settings_.transformVelocityPhysicalToLB (m.value[0]) ;
		node.uBoundary (Y) = settings_.transformVelocityPhysicalToLB (m.value[1]) ;
		node.uBoundary (Z) = settings_.transformVelocityPhysicalToLB (m.value[2]) ;
	}

	for (auto m : modificationRhoU.rhoPhysical)
	{
		GET_NODE ;

		node.rho() = settings_.transformPressurePhysicalToVolumetricMassDensityLB (m.value) ;
	}

	for (auto m : modificationRhoU.rhoBoundaryPhysical)
	{
		GET_NODE ;

		node.rhoBoundary() = settings_.transformPressurePhysicalToVolumetricMassDensityLB (m.value) ;
	}

	#undef GET_NODE
}



template class TiledLattice <D3Q19, double, StorageOnCPU, TileDataArrangement::XYZ> ;
template class TiledLattice <D3Q19, double, StorageOnCPU, TileDataArrangement::OPT_1> ;



}
