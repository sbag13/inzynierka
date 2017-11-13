#ifndef TILING_STATISTIC_HPP
#define TILING_STATISTIC_HPP

#include <cstddef>
#include <string>
#include <sstream>

#include "TileLayout.hpp"
#include "NodeType.hpp"

namespace microflow
{



// FIXME: refactorize, sort out a mess in the interface
class TilingStatistic
{
	public:

		typedef TileLayout<StorageOnCPU>  TileLayoutType ;
		typedef TileLayout<StorageOnCPU>::NonEmptyTile TileType ;

		TilingStatistic() ;

		// TODO: maybe tileLayout should be local attribute ? TilingStatistic is for single tile layout.
		void addTile( const TileType & tile, const TileLayoutType & tileLayout ) ;

		static constexpr unsigned getTileEdge() {return DEFAULT_3D_TILE_EDGE ;} ;
		static constexpr unsigned getNodesPerTile()
		{ return TileType::getNNodesPerTile() ; } ;

		// FIXME: make private, leave only computeStatistics() ?
		size_t getNSolidNodesInTiles() const ;
		size_t getNSolidNodesInTotal() const ;
		size_t getNFluidNodes() const ;
		size_t getNBoundaryNodes() const ;
		size_t getNUnknownNodes() const ;
		size_t getNNonSolidNodes() const ;

		size_t computeNTotalNodes() const ;
		double computeBoundaryToFluidNodesRatio() const ;

		size_t getNNonEmptyTiles() const ;
		size_t getNEmptyTiles() const ;
		size_t getNTotalTiles() const ;
		size_t getNNodesInNonEmptyTiles() const ;

		double computeAverageTileUtilisation() const ;
		double computeNonEmptyTilesFactor() const ;
		double computeGeometryDensity() const ;

		void setTileGridSize( Size size ) ;
		Size getTileGridSize() const ;
		Size getNodeGridSize() const ;

		//FIXME: should these be private ? We have addTile() now.
		void setNTotalTiles( size_t nTotalTiles ) ;
		void addNode ( const NodeType & nodeType ) ;
		void increaseNonemptyTilesCounter() ;

		std::string computeStatistics() const ;


	private:		

		std::size_t nSolidNodesInTiles_ ;
		std::size_t nFluidNodes_ ;
		std::size_t nBoundaryNodes_ ;
		std::size_t nUnknownNodes_ ;

		std::size_t nTotalTiles_ ;
		std::size_t nNonEmptyTiles_ ;

		Size tileGridSize_ ;
} ;




}
#endif
