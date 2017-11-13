#include "gtest/gtest.h"

#include <sstream>

#include "ReaderVtk.hpp"
#include "Writer.hpp"
#include "PerformanceMeter.hpp"
#include "CheckpointSettings.hpp"

#include "TestPath.hpp"
#include "TiledLatticeTest.hpp"
#include "NodeLayoutTest.hpp"
#include "NodeFromTileTest.hpp"

#include <vtkCellType.h>



using namespace microflow ;
using namespace std ;




void testReaderVtk (NodeLayout & nodeLayout, string testName)
{
	PerformanceMeter pm ;

	//TODO: Duplicated from WriterTest.cc: TEST (Sparse, save)
	pm.start() ;
	TileLayout <StorageOnCPU> tileLayout (nodeLayout) ;
	pm.stop() ;
	logger << "TileLayout building: " << pm.generateSummary() << "\n" ;
	pm.clear() ;


	pm.start() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;
	pm.stop() ;
	logger << "ExpandedNodeLayout building: " << pm.generateSummary() << "\n" ;
	pm.clear() ;


	typedef TiledLattice <D3Q19, double, StorageOnCPU, TileDataArrangement::XYZ> 
		TLattice ;

	pm.start() ;
	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;
	pm.stop() ;
	logger << "TiledLattice building: " << pm.generateSummary() << "\n" ;
	pm.clear() ;


	fillWithConsecutiveValues (tiledLattice) ;
	
	/*
		In checkpoint file we store only one copy of f_i functions.
		During read both f and fPost copies are set to identical values.
		Thus we need to have f nad fPost the same to avoid errors while
		comparing tiledLattice with tLattice2.
	*/
	tiledLattice.forEachNode
	(
		[&] (typename TLattice::TileType::DefaultNodeType & node,
				 Coordinates & globCoord)
		{
			for (Direction::DirectionIndex q=0 ; 
					 q < TLattice::LatticeArrangementType::getQ() ;
					 q++)
			{
				node.fPost (q) = node.f (q) ;
			}
		}
	) ;
	tiledLattice.setValidCopyIDToF() ;


	/*
						Structured grid.
	*/
	{
		string fileName = testOutDirectory + testName + ".vti" ;
		auto writer = vtkSmartPointer <WriterVtkImage>::New() ;

		writer->SetFileName (fileName.c_str()) ;
		writer->SetDataModeToBinary() ; // Ascii does not support NANs.

		writer->write (tiledLattice, CheckpointSettings()) ;

		/*
			 Reading.
		 */
		TLattice tLattice2 (tileLayout, expandedNodeLayout, Settings()) ;

		auto reader = vtkSmartPointer <ReaderVtkImage>::New() ;

		reader->SetFileName (fileName.c_str()) ;
		ASSERT_NO_THROW (reader->read (tLattice2)) ;

		ASSERT_TRUE (areEqual (tiledLattice, tLattice2)) ;
		ASSERT_EQ (tiledLattice, tLattice2) ;

		stringstream ss1, ss2 ;

		ASSERT_NO_THROW (writer->registerOstream (ss1)) ;
		ASSERT_NO_THROW (writer->write (tiledLattice, CheckpointSettings())) ;
		ASSERT_NO_THROW (writer->registerOstream (ss2)) ;
		ASSERT_NO_THROW (writer->write (tLattice2, CheckpointSettings())) ;
		ASSERT_EQ (ss1.str(), ss2.str()) ;
	}

	/*
						Unstructured grid.
	*/
	{
		string fileName = testOutDirectory + testName + ".vtu" ;
		auto writer = vtkSmartPointer <WriterVtkUnstructured>::New() ;

		writer->SetFileName (fileName.c_str()) ;
		writer->SetDataModeToBinary() ;

		writer->write (tiledLattice, CheckpointSettings()) ;

		TLattice tLattice2 (tileLayout, expandedNodeLayout, Settings()) ;

		auto reader = vtkSmartPointer <ReaderVtkUnstructured>::New() ;

		reader->SetFileName (fileName.c_str()) ;

		ASSERT_NO_THROW (reader->read (tLattice2)) ;
    
		ASSERT_EQ (tiledLattice, tLattice2) ;

		stringstream ss1, ss2 ;

		ASSERT_NO_THROW (writer->registerOstream (ss1)) ;
		ASSERT_NO_THROW (writer->write (tiledLattice, CheckpointSettings())) ;
		ASSERT_NO_THROW (writer->registerOstream (ss2)) ;
		ASSERT_NO_THROW (writer->write (tLattice2, CheckpointSettings())) ;
		ASSERT_EQ (ss1.str(), ss2.str()) ;
	}
}



TEST (ReaderVtk, singleTile)
{
	auto nodeLayout = createFluidNodeLayout (4,4,4) ;

	testReaderVtk (nodeLayout, "ReaderVtk_singleTile") ;
}



TEST (ReaderVtk, 2x2x2Tiles)
{
	auto nodeLayout = createFluidNodeLayout (2*4,2*4,2*4) ;

	testReaderVtk (nodeLayout, "ReaderVtk_2x2x2Tiles") ;
}



TEST (ReaderVtk, 3x3x3Tiles_sparse)
{
	auto nodeLayout = createSolidNodeLayout (3*4,3*4,3*4) ;

	for (unsigned i=0 ; i < 3 ; i++)
	{
		Coordinates tileCorner (i*4, i*4, i*4) ;

		for (unsigned z=0 ; z < 4 ; z++)
			for (unsigned y=0 ; y < 4 ; y++)
				for (unsigned x=0 ; x < 4 ; x++)
				{
					Coordinates nodeInTileCoord (x,y,z) ;
					Coordinates globalNodeCoord = tileCorner + nodeInTileCoord ;

					nodeLayout.setNodeType (globalNodeCoord, NodeBaseType::FLUID) ;
				}
	}

	testReaderVtk (nodeLayout, "ReaderVtk_3x3x3Tiles_sparse") ;
}



TEST (ReaderVtk, simulation_3)
{
	Settings * settings = NULL ;
	EXPECT_NO_THROW 
	( 
		settings = new Settings("./test_data/cases/simulation_3_GPU")
	) ;

	NodeLayout nodeLayout 
		(
		 ColoredPixelClassificator (settings->getPixelColorDefinitionsFilePath()),
		 Image (settings->getGeometryPngImagePath()),
		 settings->getZExpandDepth()
		) ;

	testReaderVtk (nodeLayout, "ReaderVtk_simulation_3") ;
}



