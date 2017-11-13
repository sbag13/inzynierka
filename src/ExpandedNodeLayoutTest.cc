#include "gtest/gtest.h"
#include <iostream>

#include "Logger.hpp"
#include "PerformanceMeter.hpp"
#include "TiledLattice.hpp"
#include "ExpandedNodeLayout.hpp"



using namespace microflow ;
using namespace std ;



TEST (ExpandedNodeLayout, profiling)
{
	//const string caseName = "A2_1000x1000" ;
	const string caseName = "A2_2200x2200_GPU" ;
	//const string caseName = "A3_5000x4000_GPU" ;
	string casePath = "./test_data/cases/" + caseName ;

	logger << "Reading simulation from " << casePath << "\n" ;

	unique_ptr <Settings> settings ;
	PerformanceMeter pm ;

	ASSERT_NO_THROW (settings.reset (new Settings (casePath))) ;

	ColoredPixelClassificator 
		coloredPixelClassificator (settings->getPixelColorDefinitionsFilePath()) ;

	pm.start() ;
	Image image (settings->getGeometryPngImagePath()) ;
	pm.stop() ;
	logger << "Image read: " << pm.generateSummary() << "\n" ;
	pm.clear() ;

	pm.start() ;
	NodeLayout nodeLayout (coloredPixelClassificator, image, 
												 settings->getZExpandDepth()) ;
	pm.stop() ;
	logger << "NodeLayout: " << pm.generateSummary() << "\n" ;
	pm.clear() ;

	pm.start() ;
	settings->initialModify (nodeLayout) ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	if ( 1 < settings->getZExpandDepth() )
	{
		expandedNodeLayout.computeSolidNeighborMasks() ;
		expandedNodeLayout.computeNormalVectors() ;
		nodeLayout.temporaryMarkUndefinedBoundaryNodesAndCovers() ;
		nodeLayout.restoreBoundaryNodes (coloredPixelClassificator, image) ;
		expandedNodeLayout.classifyNodesPlacedOnBoundary (*settings) ;
		expandedNodeLayout.classifyPlacementForBoundaryNodes (*settings) ;
	}
	settings->finalModify (nodeLayout) ;
	//TODO: unoptimal (computed twice), but easy.
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;
	pm.stop() ;
	logger << "ExpandedNodeLayout: " << pm.generateSummary() << "\n" ;
	pm.clear() ;


	pm.start() ;
	TileLayout <StorageOnCPU> tileLayout (nodeLayout) ;
	pm.stop() ;
	logger << "TileLayout building: " << pm.generateSummary() << "\n" ;
	pm.clear() ;


	typedef TiledLattice <D3Q19, double, StorageOnCPU, TileDataArrangement::XYZ> 
		TLattice ;

	pm.start() ;
	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;
	pm.stop() ;
	logger << "TiledLattice building: " << pm.generateSummary() << "\n" ;
	pm.clear() ;
}



