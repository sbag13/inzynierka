#include "TilingStatistic.hpp"
#include "TileDefinitions.hpp"

#include <iomanip>



using namespace std ;



namespace microflow
{



TilingStatistic::
TilingStatistic() :
	nSolidNodesInTiles_ (0),
	nFluidNodes_        (0),
	nBoundaryNodes_     (0),
	nUnknownNodes_      (0),
	nTotalTiles_        (0),
	nNonEmptyTiles_     (0)
{
}



void TilingStatistic::
addTile( const TileType & tile, const TileLayoutType & tileLayout )
{
	for (auto iterator = tile.getBeginOfNodes() ;
						iterator < tile.getEndOfNodes() ;
						iterator ++ )
	{
		Coordinates coordinates = tile.unpack( iterator ) ;
		NodeType node = tileLayout.getNodeType( coordinates ) ;
		addNode( node ) ;
	}

	nNonEmptyTiles_ ++ ;
	nTotalTiles_    ++ ;
}



void TilingStatistic::
addNode( const NodeType & nodeType )
{
	NodeBaseType nodeBaseType = nodeType.getBaseType() ;

	switch (nodeBaseType)
	{
		case NodeBaseType::SOLID:
			nSolidNodesInTiles_ ++ ;
			break ;
		case NodeBaseType::FLUID:
			nFluidNodes_ ++ ;
			break ;
		case NodeBaseType::BOUNCE_BACK_2:
		case NodeBaseType::VELOCITY:
		case NodeBaseType::VELOCITY_0:
		case NodeBaseType::PRESSURE:
			nBoundaryNodes_ ++ ;
			break ;
		case NodeBaseType::MARKER:
		case NodeBaseType::SIZE:
			nUnknownNodes_ ++ ;
			break ;
	}
}



void TilingStatistic::
increaseNonemptyTilesCounter()
{
	nNonEmptyTiles_ ++ ;
}



void TilingStatistic::
setNTotalTiles( size_t nTotalTiles )
{
	assert( nTotalTiles >= nNonEmptyTiles_ ) ;
	nTotalTiles_ = nTotalTiles ;
}



double TilingStatistic::
computeNonEmptyTilesFactor() const
{
	return static_cast<double>(nNonEmptyTiles_) / nTotalTiles_ ;
}

size_t TilingStatistic::
getNSolidNodesInTiles() const
{
	return nSolidNodesInTiles_ ;
}



size_t TilingStatistic::
getNSolidNodesInTotal() const
{
	size_t nodesPerTile = DEFAULT_3D_TILE_EDGE * DEFAULT_3D_TILE_EDGE * DEFAULT_3D_TILE_EDGE ;
	size_t nSolidNodesInEmptyTiles = getNEmptyTiles() * nodesPerTile ;

	return nSolidNodesInEmptyTiles + getNSolidNodesInTiles() ;
}



size_t TilingStatistic::
getNFluidNodes() const
{
	return nFluidNodes_ ;
}



size_t TilingStatistic::
getNBoundaryNodes() const
{
	return nBoundaryNodes_ ;
}



size_t TilingStatistic::
getNUnknownNodes() const
{
	return nUnknownNodes_ ;
}



size_t TilingStatistic::
getNNonSolidNodes() const
{
	return 	getNFluidNodes()    + 
	 				getNBoundaryNodes() + 
					getNUnknownNodes()  ;
}



size_t TilingStatistic::
computeNTotalNodes() const 
{
	return getNTotalTiles() * TileType::getNNodesPerTile() ;
}



double TilingStatistic::
computeBoundaryToFluidNodesRatio() const 
{
	double nBoundaryNodes = getNBoundaryNodes() ;

	if (0 == nBoundaryNodes ) 
	{
		return 0 ;
	}

	return nBoundaryNodes / getNFluidNodes() ;
}




size_t TilingStatistic::
getNNonEmptyTiles() const
{
	return nNonEmptyTiles_ ;
}



size_t TilingStatistic::
getNTotalTiles() const
{
	return nTotalTiles_ ;
}



size_t TilingStatistic::
getNEmptyTiles() const
{
	return getNTotalTiles() - getNNonEmptyTiles() ;
}



double TilingStatistic::
computeAverageTileUtilisation() const
{
	size_t nNodesInNonEmptyTiles = getNSolidNodesInTiles() + 
																 getNNonSolidNodes() ;

	if ( 0 == getNNonSolidNodes() ) 
	{
		return 0 ;
	}

	return 
		static_cast<double>(getNNonSolidNodes()) / nNodesInNonEmptyTiles ;
}



double TilingStatistic::
computeGeometryDensity() const
{
	double totalNodes = static_cast<double>(getNTotalTiles()) * 
											TileLayout<StorageOnCPU>::NonEmptyTile::getNNodesPerTile() ;
	
	double nonSolidNodes = getNNonSolidNodes() ;

	if ( 0.0 == nonSolidNodes ) 
	{
		return 0 ;
	}

	return nonSolidNodes / totalNodes ;
}



string printPercent( double v )
{
	stringstream s ;

  s << std::fixed << std::setprecision(2) << std::setw(6) ;
	s << 100.0 * v << " %" ;

	return s.str() ;
}



string printUint( size_t u )
{
	stringstream s ;

	s << setw(10) << u ;

	return s.str() ;
}



string printFraction( double f )
{
	stringstream s ;
	
	s << setw(10) << setprecision(6) << showpoint << f ;

	return s.str() ;
}



string printUintPercent( size_t value, size_t maxValue )
{
	stringstream s ;

	s << printUint( value ) ;

	double nodesPercent = static_cast<double>(value) / maxValue ;
	s << " ( " << printPercent(nodesPercent) << " )" ;

	return s.str() ;
}



string printGridSize( Size size )
{
	stringstream s ;

	s << size.getWidth() << " x " << size.getHeight() ;
	s << " x " << size.getDepth() ;

	return s.str() ;
}



std::string TilingStatistic::
computeStatistics() const
{
	stringstream str ;

	size_t totalNodes = computeNTotalNodes() ;


	str << "Geometry statistics:\n" ;

	str << "    total nodes                 = " ;
	str << printUint( computeNTotalNodes() ) ;
	str << " ( " << printGridSize( getNodeGridSize() ) << " ) \n" ;

	str << "    solid nodes                 = " ;
	str << printUintPercent( getNSolidNodesInTotal(), totalNodes) << "\n" ;

	str << "    fluid nodes                 = " ;
	str << printUintPercent( getNFluidNodes(), totalNodes) << "\n" ;

	str << "    boundary nodes              = " ;
	str << printUintPercent( getNBoundaryNodes(), totalNodes) << "\n" ;

	str << "    unknown nodes               = " ;
	str << printUintPercent( getNUnknownNodes(), totalNodes) << "\n" ;

	str << "    geometry density            = " ;
	str << printFraction( computeGeometryDensity() ) << " \n" ;

	str << "    boundary to non-solid nodes = " ;
	str << printFraction( computeBoundaryToFluidNodesRatio() ) << " \n" ;



	str << "Geometry partitioned UNIFORMLY, statistics:\n" ;

	str << "    tile edge                   = " ;
	str << printUint( getTileEdge() ) << " nodes \n" ;

	str << "    nodes per tile              = " ;
	str << printUint( getNodesPerTile() ) << " nodes \n" ;

	str << "    total tiles                 = " ; 
	str << printUint( getNTotalTiles() ) ;
	str << " ( " << printGridSize( getTileGridSize() ) << " ) \n" ;

	str << "    non-empty tiles             = " ; 
	str << printUintPercent( getNNonEmptyTiles(), getNTotalTiles()) << "\n" ;

	str << "    nodes in non-empty tiles    = " ;
	str << printUint( getNNodesInNonEmptyTiles() ) << " \n" ;

	str << "    average tile utilization (eta_t) = " ;
	str << printFraction( computeAverageTileUtilisation() ) << " \n" ;

	str << "    memory overhead from tile utilization (Delta_M_eta_t) = " ;
	double deltaMEtaT = getNSolidNodesInTiles() ;
	deltaMEtaT /= getNNonSolidNodes() ;
	str << printFraction( deltaMEtaT ) << " \n" ;

	return str.str() ;
}



void TilingStatistic::
setTileGridSize( Size size ) 
{
	tileGridSize_ = size ;
}



Size TilingStatistic::
getTileGridSize() const 
{
	return tileGridSize_ ;
}



Size TilingStatistic::
getNodeGridSize() const 
{
	Size gridSize = getTileGridSize() ;

	return Size( gridSize.getWidth() * getTileEdge(),
							 gridSize.getHeight() * getTileEdge(),
							 gridSize.getDepth() * getTileEdge()
						 ) ;
}



size_t TilingStatistic::
getNNodesInNonEmptyTiles() const
{
	return getNNonEmptyTiles() * getNodesPerTile() ;
}



}
