#include "TileLayout.hpp"
#include "TileDefinitions.hpp"
#include "Logger.hpp"
#include "NodeLayoutWriter.hpp"
#include "TilingStatistic.hpp"

#include <fstream>



namespace microflow
{



TileLayout<StorageOnCPU>::TileLayout( const NodeLayout & nodeLayout )
: nodeLayout_( nodeLayout )
{
	size_t widthInNodes  = nodeLayout.getSize().getWidth () ;
	size_t heightInNodes = nodeLayout.getSize().getHeight() ;
	size_t depthInNodes  = nodeLayout.getSize().getDepth () ;

#define EXTEND_TO_FULL_TILE(size)                                        \
	if (0 != ((size) % DEFAULT_3D_TILE_EDGE) )                             \
	{                                                                      \
		logger << "WARNING: " << #size << " = " << size ;                    \
		logger << " is not divisible by " << DEFAULT_3D_TILE_EDGE ;          \
		size = (size / DEFAULT_3D_TILE_EDGE  + 1) * DEFAULT_3D_TILE_EDGE ;   \
		logger << ", extending to " << size << "\n" ;                        \
	}

	EXTEND_TO_FULL_TILE( widthInNodes  ) 
	EXTEND_TO_FULL_TILE( heightInNodes )
	EXTEND_TO_FULL_TILE( depthInNodes  )

#undef EXTEND_TO_FULL_TILE

	size_ = Size( widthInNodes  / DEFAULT_3D_TILE_EDGE ,
								heightInNodes / DEFAULT_3D_TILE_EDGE ,
								depthInNodes  / DEFAULT_3D_TILE_EDGE ) ;

	Size newNodeLayoutSize( widthInNodes, heightInNodes, depthInNodes ) ;
	nodeLayout_.resizeWithContent( newNodeLayoutSize ) ;

	tileMap_.resize( size_, EMPTY_TILE ) ;

	for (size_t tileX = 0 ; tileX < size_.getWidth() ; tileX ++ )
		for (size_t tileY = 0 ; tileY < size_.getHeight() ; tileY ++ )
			for (size_t tileZ = 0 ; tileZ < size_.getDepth() ; tileZ ++ )
			{
				size_t tileX0 = tileX * DEFAULT_3D_TILE_EDGE ;
				size_t tileY0 = tileY * DEFAULT_3D_TILE_EDGE ;
				size_t tileZ0 = tileZ * DEFAULT_3D_TILE_EDGE ;

				if (not isTileEmpty(tileX0, tileY0, tileZ0) )
				{
					tilesX0_.push_back( tileX0 ) ;
					tilesY0_.push_back( tileY0 ) ;
					tilesZ0_.push_back( tileZ0 ) ;

					Coordinates c( tileX, tileY, tileZ ) ;
					tileMap_.setValue(c, tilesX0_.size() - 1) ;
				}
			}


	auto tStat = computeTilingStatistic() ;

	double nNodesInTiles = tStat.getNNodesInNonEmptyTiles() ;
	double nBoundaryNodesInTiles = tStat.getNBoundaryNodes() ;
	double nFluidNodesInTiles = tStat.getNFluidNodes() ;
	double nSolidNodesInTiles = tStat.getNSolidNodesInTiles() ;

	logger << "\n\nTile layout statistics:\n\n"
				 << "Geometry density       : " 
				 << tStat.computeGeometryDensity() << "\n"
				 << "Avg. tile util.        : " 
				 << tStat.computeAverageTileUtilisation() << "\n"
				 << "Fluid nodes            : " 
				 << tStat.getNFluidNodes() 
				 << " (" << nFluidNodesInTiles / nNodesInTiles << ")\n"
				 << "Boundary nodes         : "
				 << tStat.getNBoundaryNodes() 
				 << " (" << nBoundaryNodesInTiles / nNodesInTiles << ")\n"
				 << "Solid nodes in tiles   : "
				 << tStat.getNSolidNodesInTiles() 
				 << " (" << nSolidNodesInTiles / nNodesInTiles << ")\n"
				 << "All nodes in tiles     : " 
				 << tStat.getNNodesInNonEmptyTiles() << "\n"
				 << "Non-empty tiles        : " 
				 << tStat.getNNonEmptyTiles() << "\n"
				 << "Boundary to fluid ratio : " 
				 << tStat.computeBoundaryToFluidNodesRatio() << "\n"
				 << "\n\n" ;
}



bool TileLayout<StorageOnCPU>::
isTileEmpty(size_t tileX0, size_t tileY0, size_t tileZ0) const
{
	for (size_t z = tileZ0 ; z < tileZ0 + DEFAULT_3D_TILE_EDGE ; z++)
		for (size_t y = tileY0 ; y < tileY0 + DEFAULT_3D_TILE_EDGE ; y++)
			for (size_t x = tileX0 ; x < tileX0 + DEFAULT_3D_TILE_EDGE ; x++)
			{
				if (not nodeLayout_.getNodeType(x, y, z).isSolid() ) 
					return false ;
			}

	return true ;
}



size_t TileLayout<StorageOnCPU>::
computeNoTilesTotal() const
{
	return size_.computeVolume() ;
}



void TileLayout<StorageOnCPU>::
saveToVolFile( std::string fileName ) const
{
	NodeLayout layout = generateLayoutWithMarkedTiles() ;
	NodeLayoutWriter().saveToVolFile( layout, fileName ) ;
}



NodeLayout TileLayout<StorageOnCPU>::
generateLayoutWithMarkedTiles() const
{
	NodeLayout layout( nodeLayout_ ) ;

	for (size_t t=0 ; t < tilesX0_.size() ; t++)
	{
		size_t tX0 = tilesX0_[t] ;
		size_t tY0 = tilesY0_[t] ;
		size_t tZ0 = tilesZ0_[t] ;

		const size_t tE = DEFAULT_3D_TILE_EDGE - 1 ;
		
		for (size_t i=0 ; i < DEFAULT_3D_TILE_EDGE ; i++)
		{
			layout.setNodeType( tX0 + i , tY0     , tZ0     , layoutBorderNode ) ;
			layout.setNodeType( tX0 + i , tY0 + tE, tZ0     , layoutBorderNode ) ;
			layout.setNodeType( tX0 + i , tY0     , tZ0 + tE, layoutBorderNode ) ;
			layout.setNodeType( tX0 + i , tY0 + tE, tZ0 + tE, layoutBorderNode ) ;

			layout.setNodeType( tX0     , tY0 + i , tZ0     , layoutBorderNode ) ;
			layout.setNodeType( tX0 + tE, tY0 + i , tZ0     , layoutBorderNode ) ;
			layout.setNodeType( tX0     , tY0 + i , tZ0 + tE, layoutBorderNode ) ;
			layout.setNodeType( tX0 + tE, tY0 + i , tZ0 + tE, layoutBorderNode ) ;

			layout.setNodeType( tX0     , tY0     , tZ0 + i , layoutBorderNode ) ;
			layout.setNodeType( tX0 + tE, tY0     , tZ0 + i , layoutBorderNode ) ;
			layout.setNodeType( tX0     , tY0 + tE, tZ0 + i , layoutBorderNode ) ;
			layout.setNodeType( tX0 + tE, tY0 + tE, tZ0 + i , layoutBorderNode ) ;
		}
	}
	
	// TODO: useful for images generated by qvox, but is it the correct way ?
	for (auto it = layout.begin() ; it < layout.end() ; it++)
	{
		it->setBoundaryDefinitionIndex(0) ;
	}

	return layout ;
}




TilingStatistic TileLayout<StorageOnCPU>::
computeTilingStatistic() const
{
	TilingStatistic totalStatistic ;
	
	for (auto iterator = getBeginOfNonEmptyTiles() ;
						iterator < getEndOfNonEmptyTiles() ;
						iterator ++ )
	{
		totalStatistic.addTile( getTile(iterator), *this ) ;
	}
	
	totalStatistic.setNTotalTiles( computeNoTilesTotal() ) ;
	totalStatistic.setTileGridSize( size_ ) ;
	return totalStatistic ;
}



}
