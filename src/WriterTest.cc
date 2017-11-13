#include "gtest/gtest.h"

#include <sstream>

#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkStructuredPoints.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkCellType.h>

#include "Writer.hpp"
#include "PerformanceMeter.hpp"
#include "ExpandedNodeLayout.hpp"
#include "CheckpointSettings.hpp"
#include "BaseIO.hpp"

#include "TiledLatticeTest.hpp"
#include "TestPath.hpp"



using namespace microflow ;
using namespace std ;



TEST (Writer, t1)
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(5, 5, 5) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <D3Q19, double, StorageOnCPU, TileDataArrangement::XYZ> TLattice ;

	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;
	
	typename TLattice::Iterator ti ;

	EXPECT_NO_THROW (ti = tiledLattice.getBeginOfTiles()) ;

	EXPECT_NO_THROW  (tiledLattice.getTile (ti)) ;
	EXPECT_ANY_THROW (tiledLattice.getTile (++ti)) ;

	auto writer = Writer <TLattice::LatticeArrangementType,
												TLattice::DataTypeType,
												TLattice::LatticeDataArrangement> (tiledLattice) ;

	cout << "Estimated data size for structured grid  : " 
						<< writer.estimateDataSizeForStructuredGrid (CheckpointSettings()) 
						<< "\n" ;
	cout << "Estimated data size for unstructured grid: " 
						<< writer.estimateDataSizeForUnstructuredGrid (CheckpointSettings()) 
						<< "\n" ;

	#warning "Untested"
}


// Look at http://www.paraview.org/Wiki/VTK/Examples/Cxx/ImageData/IterateImageData
//
//	TODO: Make it as a WriterVtkImage::buildStructuredGrid() method.
//
template< class TiledLattice, class Settings >
static
inline void
saveVtkToStream
( 
	const TiledLattice & tiledLattice, 
	const Settings & settings,
	ostream & os
)
{
	// TODO: reconsider, if we need saving only node base types without placement modifiers.
	const bool shouldSaveBaseNodeTypes = settings.shouldSaveNodes() ;
	const bool shouldSaveOtherNodeTypes = settings.shouldSaveNodes() ;

	typedef typename TiledLattice::DataTypeType DataType ;
	typedef typename TiledLattice::LatticeArrangementType LatticeArrangement ;

	const Size geometrySize = tiledLattice.getTileLayout().getNodeLayout().getSize() ;

	auto savedData = vtkSmartPointer< vtkStructuredPoints >::New() ;
	savedData->SetOrigin( 0,0,0 ) ;
	savedData->SetSpacing( 1,1,1 ) ; // dxPhys, dxPhys, dxPhys
	savedData->SetDimensions( geometrySize.getX(), geometrySize.getY(), geometrySize.getZ() ) ;

	const int nPoints = savedData->GetNumberOfPoints() ;

#define INITIALIZE_VTK_ARRAY( name, nComponents, nTuples, defaultValue)                 \
	name->SetName( #name ) ;                                                              \
	name->SetNumberOfComponents( (nComponents) ) ;                                        \
	name->SetNumberOfTuples( (nTuples) ) ;                                                \
	for (vtkIdType tupleIdx=0 ; tupleIdx < name->GetNumberOfTuples() ; tupleIdx ++)       \
		for (int componentIdx=0 ; componentIdx < nComponents ; componentIdx ++)             \
		{                                                                                   \
			name->SetComponent (tupleIdx, componentIdx, static_cast<double>(defaultValue)) ;  \
		}

#define DEFINE_VTK_ARRAY( name, nComponents, nTuples, defaultValue, shouldSave )        \
	auto name = vtkSmartPointer< typename VtkTypes<DataType>::ArrayType >::New() ;        \
	if ( shouldSave ) { INITIALIZE_VTK_ARRAY( name, nComponents, nTuples, defaultValue) }

#define DEFINE_VTK_ARRAY_3D( name, nTuples, shouldSave )            \
	DEFINE_VTK_ARRAY( name, 3, (nTuples), NAN, (shouldSave) ) ;       \
	if (shouldSave) {                                                 \
	name->SetComponentName( 0, "x" ) ;                                \
	name->SetComponentName( 1, "y" ) ;                                \
	name->SetComponentName( 2, "z" ) ; }

	DEFINE_VTK_ARRAY_3D( velocity          , nPoints, settings.shouldSaveVelocityPhysical() ) ;
	DEFINE_VTK_ARRAY_3D( velocityLB        , nPoints, settings.shouldSaveVelocityLB() ) ;
	DEFINE_VTK_ARRAY_3D( velocityT0LB      , nPoints, settings.shouldSaveVelocityLB() ) ;
	DEFINE_VTK_ARRAY_3D( boundaryVelocity  , nPoints, settings.shouldSaveVelocityPhysical() ) ;
	DEFINE_VTK_ARRAY_3D( boundaryVelocityLB, nPoints, settings.shouldSaveVelocityLB() ) ;

	DEFINE_VTK_ARRAY (rhoLB      , 1, nPoints, NAN, settings.shouldSaveVolumetricMassDensityLB()) ;
	DEFINE_VTK_ARRAY (rhoT0LB    , 1, nPoints, NAN, settings.shouldSaveVolumetricMassDensityLB()) ;
	DEFINE_VTK_ARRAY (boundaryRho, 1, nPoints, NAN, settings.shouldSaveVolumetricMassDensityLB()) ;
	DEFINE_VTK_ARRAY (pressure   , 1, nPoints, NAN, settings.shouldSavePressurePhysical()) ;

	auto nodeBaseType = vtkSmartPointer< vtkUnsignedCharArray >::New() ;
	if ( shouldSaveBaseNodeTypes )
	{
		INITIALIZE_VTK_ARRAY (nodeBaseType, 1, nPoints, NodeBaseType::SOLID) ;
	}

	auto nodePlacementModifier = vtkSmartPointer< vtkUnsignedCharArray >::New() ;
	if ( shouldSaveOtherNodeTypes )
	{
		INITIALIZE_VTK_ARRAY( nodePlacementModifier, 1, nPoints, PlacementModifier::NONE) ;
	}

#undef DEFINE_VTK_ARRAY_3D
#undef DEFINE_VTK_ARRAY
#undef INITIALIZE_VTK_ARRAY

	constexpr unsigned nQ = LatticeArrangement::getQ() ;
	vector< vtkSmartPointer< typename VtkTypes<DataType>::ArrayType > > fArray( nQ ) ;
	vector< vtkSmartPointer< typename VtkTypes<DataType>::ArrayType > > fPostArray( nQ ) ;
	if ( settings.shouldSaveMassFlowFractions() )
	{
		for (Direction::DirectionIndex q=0 ; q < nQ ; q++)
		{
			fArray[ q ] = vtkSmartPointer< typename VtkTypes<DataType>::ArrayType >::New() ;
			fArray[ q ]->SetNumberOfComponents( 1 ) ;
			fArray[ q ]->SetNumberOfTuples( nPoints ) ;
			string fArrayName = buildFArrayName< LatticeArrangement >("f", q) ;
			fArray[ q ]->SetName( fArrayName.c_str() ) ;

			fPostArray[ q ] = vtkSmartPointer< typename VtkTypes<DataType>::ArrayType >::New() ;
			fPostArray[ q ]->SetNumberOfComponents( 1 ) ;
			fPostArray[ q ]->SetNumberOfTuples( nPoints ) ;
			fArrayName = buildFArrayName< LatticeArrangement >("fPost", q) ;
			fPostArray[ q ]->SetName( fArrayName.c_str() ) ;

			for (vtkIdType tupleIdx=0 ; tupleIdx < nPoints ; tupleIdx ++)
			{
				fArray    [q]->SetComponent (tupleIdx, 0, NAN) ;
				fPostArray[q]->SetComponent (tupleIdx, 0, NAN) ;
			}

		}
	}

	for (auto t = tiledLattice.getBeginOfTiles() ; t < tiledLattice.getEndOfTiles() ; t++)
	{
		typename TiledLattice::TileType tile = tiledLattice.getTile( t ) ;
		auto u = tile.u() ;
		auto uT0 = tile.uT0() ;
		auto uBoundary   = tile.uBoundary() ;
		auto rhoBoundary = tile.rhoBoundary() ;
		auto nodeTypes   = tile.getNodeTypes() ;

		constexpr unsigned edge = TiledLattice::TileType::getNNodesPerEdge() ;
		for (unsigned tz=0 ; tz < edge ; tz++ )
			for (unsigned ty=0 ; ty < edge ; ty++ )
				for (unsigned tx=0 ; tx < edge ; tx++ )
				{
					Coordinates globalCoordinates = tile.getCornerPosition() + Coordinates( tx,ty,tz ) ;
					const double gx = globalCoordinates.getX() ;
					const double gy = globalCoordinates.getY() ;
					const double gz = globalCoordinates.getZ() ;
					vtkIdType pointId = savedData->FindPoint( gx,gy,gz ) ;

					double uVec [3] ;

					if ( settings.shouldSaveVelocityLB() )
					{
						uVec[0] = u[X][tz][ty][tx] ;
						uVec[1] = u[Y][tz][ty][tx] ;
						uVec[2] = u[Z][tz][ty][tx] ;

						velocityLB->SetTupleValue( pointId, uVec ) ;

						uVec[0] = uT0[X][tz][ty][tx] ;
						uVec[1] = uT0[Y][tz][ty][tx] ;
						uVec[2] = uT0[Z][tz][ty][tx] ;

						velocityT0LB->SetTupleValue( pointId, uVec ) ;

						uVec[0] = uBoundary[X][tz][ty][tx] ;
						uVec[1] = uBoundary[Y][tz][ty][tx] ;
						uVec[2] = uBoundary[Z][tz][ty][tx] ;

						boundaryVelocityLB->SetTupleValue( pointId, uVec ) ;
					}

					if ( settings.shouldSaveVelocityPhysical() )
					{
						uVec[0] = settings.transformVelocityLBToPhysical( u[X][tz][ty][tx] ) ;
						uVec[1] = settings.transformVelocityLBToPhysical( u[Y][tz][ty][tx] ) ;
						uVec[2] = settings.transformVelocityLBToPhysical( u[Z][tz][ty][tx] ) ;
						
						velocity->SetTupleValue( pointId, uVec ) ;

						uVec[0] = settings.transformVelocityLBToPhysical( uBoundary[X][tz][ty][tx] ) ;
						uVec[1] = settings.transformVelocityLBToPhysical( uBoundary[Y][tz][ty][tx] ) ;
						uVec[2] = settings.transformVelocityLBToPhysical( uBoundary[Z][tz][ty][tx] ) ;

						boundaryVelocity->SetTupleValue( pointId, uVec ) ;
					}

					if ( settings.shouldSaveVolumetricMassDensityLB() )
					{
						double r = tile.rho()[tz][ty][tx] ;
						if ( nodeTypes[tz][ty][tx] == NodeBaseType::BOUNCE_BACK_2 )
						{
							// FIXME: Mean_Rho NOT IMPLEMENTED
							// tile.rho()[tz][ty][tx] = r ; // Probably needed for pressure
						}
						rhoLB->SetTupleValue( pointId, &r ) ;

						double rhoB = rhoBoundary[tz][ty][tx] ;
						boundaryRho->SetTupleValue( pointId, &rhoB ) ;

						double rhoT0 = tile.rho0()[tz][ty][tx] ;
						rhoT0LB->SetTupleValue (pointId, &rhoT0) ;
					}

					if ( settings.shouldSavePressurePhysical() )
					{
						double p = tile.rho()[tz][ty][tx] ;
						p = settings.transformVolumetricMassDensityLBToPressurePhysical( p ) ;
						pressure->SetTupleValue( pointId, &p ) ;
					}
					
					if ( shouldSaveBaseNodeTypes )
					{
						unsigned char nt = static_cast<unsigned char>(nodeTypes[tz][ty][tx].getBaseType() ) ;
						nodeBaseType->SetTupleValue( pointId, &nt ) ;
					}

					if ( shouldSaveOtherNodeTypes )
					{
						unsigned char nt = 
							static_cast<unsigned char>(nodeTypes[tz][ty][tx].getPlacementModifier() ) ;
						nodePlacementModifier->SetTupleValue( pointId, &nt ) ;
					}

					if ( settings.shouldSaveMassFlowFractions() )
					{
						for (unsigned q=0 ; q < nQ ; q++)
						{
							auto   d = LatticeArrangement::c[ q ] ;
							double f = tile.f( d )[tz][ty][tx] ;
							fArray[ q ]->SetTupleValue( pointId, &f ) ;
							f = tile.fPost( d )[tz][ty][tx] ;
							fPostArray[ q ]->SetTupleValue( pointId, &f ) ;
						}
					}

				}
	}
	

	if ( shouldSaveBaseNodeTypes )
	{
		savedData->GetPointData()->AddArray( nodeBaseType ) ;
	}
	if ( shouldSaveOtherNodeTypes )
	{
		savedData->GetPointData()->AddArray( nodePlacementModifier ) ;
	}
	if ( settings.shouldSaveVelocityLB() )
	{
		savedData->GetPointData()->AddArray( boundaryVelocityLB ) ;
		savedData->GetPointData()->AddArray( velocityLB ) ;
		savedData->GetPointData()->AddArray( velocityT0LB ) ;
	}
	if ( settings.shouldSaveVelocityPhysical() )
	{
		savedData->GetPointData()->AddArray( boundaryVelocity ) ;
		savedData->GetPointData()->AddArray( velocity   ) ;
	}
	if ( settings.shouldSaveVolumetricMassDensityLB() )
	{
		savedData->GetPointData()->AddArray (boundaryRho) ;
		savedData->GetPointData()->AddArray (rhoLB) ;
		savedData->GetPointData()->AddArray (rhoT0LB) ;
	}
	if ( settings. shouldSavePressurePhysical() )
	{
		savedData->GetPointData()->AddArray( pressure ) ;
	}
	if ( settings.shouldSaveMassFlowFractions() )
	{
		for (unsigned q=0 ; q < nQ ; q++)
		{
			savedData->GetPointData()->AddArray( fArray[ q ] ) ;
		}
	}

	auto writer = vtkSmartPointer <WriterVtkImage>::New() ;
	writer->registerOstream (os) ;
	writer->SetCompressorTypeToNone() ;
	writer->SetDataModeToAscii() ;
  writer->SetInput (savedData) ;
  writer->Write();	
}



TEST (WriterVtkImage, t1)
{
	TileLayout<StorageOnCPU> tileLayout = generateSingleTileLayout(4, 4, 4) ;

	NodeLayout nodeLayout = tileLayout.getNodeLayout() ;
	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	expandedNodeLayout.computeSolidNeighborMasks() ;
	expandedNodeLayout.computeNormalVectors() ;

	typedef TiledLattice <D3Q19, double, StorageOnCPU, TileDataArrangement::XYZ> TLattice ;

	TLattice tiledLattice (tileLayout, expandedNodeLayout, Settings()) ;
	
	typename TLattice::Iterator ti ;

	EXPECT_NO_THROW (ti = tiledLattice.getBeginOfTiles()) ;

	EXPECT_NO_THROW  (tiledLattice.getTile (ti)) ;
	EXPECT_ANY_THROW (tiledLattice.getTile (++ti)) ;


	stringstream origStream ;

	saveVtkToStream (tiledLattice, CheckpointSettings(), origStream) ;


	stringstream myStream ;

  auto myWriter = vtkSmartPointer <WriterVtkImage>::New() ;
	myWriter->registerOstream (myStream) ;
	myWriter->SetDataModeToAscii() ;
	myWriter->SetCompressorTypeToNone () ;
	myWriter->write (tiledLattice, CheckpointSettings()) ;

	cout << "\nOriginal stream:\n\n" << origStream.str() << "\n" ;
	cout << "\nMy stream:\n\n" << myStream.str() << "\n" ;

	EXPECT_EQ (origStream.str(), myStream.str()) ;
}



TEST (WriterVtkImage, DISABLED_perf1)
{
	const unsigned width = 200 ;
	const unsigned height = 200 ;
	const unsigned depth = 200 ;

	const Size size (width,height,depth) ;

	PerformanceMeter pm ;
	
	pm.start() ;
	NodeLayout nodeLayout (size) ;
	pm.stop() ;
	logger << "NodeLayout creation: " << pm.generateSummary() << "\n" ;
	pm.clear() ;


	pm.start() ;
	for (unsigned z=0 ; z < depth ; z++)
		for (unsigned y=0 ; y < height ; y++)
			for (unsigned x=0 ; x < width ; x++)
			{
				nodeLayout.setNodeType (x,y,z, NodeBaseType::FLUID) ;
			}
	pm.stop() ;
	logger << "NodeLayout filling: " << pm.generateSummary() << "\n" ;
	pm.clear() ;

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


	auto writer = vtkSmartPointer <WriterVtkImage>::New() ;
	writer->SetDataModeToBinary() ;

	stringstream ss ;
	ss << width << "x" << height << "x" << depth ;
	string fName ;


	pm.start() ;
	
	fName = testOutDirectory + "perf1_" + ss.str() + "_compressed.vti" ;
	logger << "Writing " << fName << "\n" ;
	writer->SetFileName (fName.c_str()) ;
	writer->SetCompressorTypeToZLib() ;
	writer->write (tiledLattice, CheckpointSettings()) ;

	pm.stop() ;
	logger << "Writing compressed: " << pm.generateSummary() << "\n" ;
	pm.clear() ;
}



void readAndSaveVtk (const string caseName, bool shouldSaveStructured)
{
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


	string fNameBase = testOutDirectory + caseName + "." ;
	string fName ;


	if (shouldSaveStructured)
	{
		auto writerStructured = vtkSmartPointer <WriterVtkImage>::New() ;
		writerStructured->SetDataModeToBinary() ;
		writerStructured->SetCompressorTypeToZLib() ;

		pm.start() ;

		fName = fNameBase + writerStructured->GetDefaultFileExtension() ;
		logger << "Writing " << fName << "\n" ;
		writerStructured->SetFileName (fName.c_str()) ;
		writerStructured->write (tiledLattice, CheckpointSettings()) ;

		pm.stop() ;
		logger << "Writing structured: " << pm.generateSummary() << "\n" ;
		pm.clear() ;
	}
	
	auto writerUnstructured = vtkSmartPointer <WriterVtkUnstructured>::New() ;
	writerUnstructured->SetDataModeToBinary() ;
	writerUnstructured->SetCompressorTypeToZLib() ;

	pm.start() ;
	
	fName = fNameBase + writerUnstructured->GetDefaultFileExtension() ;
	logger << "Writing " << fName << "\n" ;
	writerUnstructured->SetFileName (fName.c_str()) ;
	writerUnstructured->write (tiledLattice, CheckpointSettings()) ;

	pm.stop() ;
	logger << "Writing unstructured: " << pm.generateSummary() << "\n" ;
	pm.clear() ;
}



TEST (A2_1000x1000, save)
{
	readAndSaveVtk ("A2_1000x1000", true) ;
}



TEST (A2_2200x2200_GPU, save)
{
	readAndSaveVtk ("A2_2200x2200_GPU", false) ;
}



TEST (A3_5000x4000_GPU, DISABLED_save)
{
	readAndSaveVtk ("A3_5000x4000_GPU", false) ;
}



void build_2T_angle (NodeLayout & nodeLayout)
{
	const unsigned width = 4 ;
	const unsigned height = 4 ;
	const unsigned depth = 8 ;

	Size size (width,height,depth) ;

	nodeLayout.resizeWithContent (size) ;

	for (unsigned z=0 ; z < depth ; z++)
		for (unsigned y=0 ; y < height ; y++)
			for (unsigned x=0 ; x < width ; x++)
			{
				nodeLayout.setNodeType (x,y,z, NodeBaseType::SOLID) ;
			}

	nodeLayout.setNodeType (1,1,0, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (1,2,0, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,1,0, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,2,0, NodeBaseType::FLUID) ;

	nodeLayout.setNodeType (1,1,1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (1,2,1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,1,1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,2,1, NodeBaseType::FLUID) ;

	nodeLayout.setNodeType (1,1,2, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (1,2,2, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,1,2, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,2,2, NodeBaseType::FLUID) ;

	nodeLayout.setNodeType (0,1,1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (0,2,1, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (0,1,2, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (0,2,2, NodeBaseType::FLUID) ;

	nodeLayout.setNodeType (1,1,0+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (1,2,0+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,1,0+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,2,0+4, NodeBaseType::FLUID) ;

	nodeLayout.setNodeType (1,1,1+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (1,2,1+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,1,1+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,2,1+4, NodeBaseType::FLUID) ;

	nodeLayout.setNodeType (1,1,2+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (1,2,2+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,1,2+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (2,2,2+4, NodeBaseType::FLUID) ;

	nodeLayout.setNodeType (0,1,1+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (0,2,1+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (0,1,2+4, NodeBaseType::FLUID) ;
	nodeLayout.setNodeType (0,2,2+4, NodeBaseType::FLUID) ;
}



void buildCross (NodeLayout & nodeLayout, Coordinates origin)
{
	auto Fluid = NodeBaseType::FLUID ;

	for (unsigned z=0 ; z < 4 ; z++)
	{
		nodeLayout.setNodeType (origin + Coordinates(1,1,z), Fluid) ;
		nodeLayout.setNodeType (origin + Coordinates(1,2,z), Fluid) ;
		nodeLayout.setNodeType (origin + Coordinates(2,1,z), Fluid) ;
		nodeLayout.setNodeType (origin + Coordinates(2,2,z), Fluid) ;
	}

	nodeLayout.setNodeType (origin + Coordinates(0,1,1), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(0,1,2), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(0,2,1), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(0,2,2), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(3,1,1), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(3,1,2), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(3,2,1), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(3,2,2), Fluid) ;

	nodeLayout.setNodeType (origin + Coordinates(1,0,1), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(1,0,2), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(2,0,1), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(2,0,2), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(1,3,1), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(1,3,2), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(2,3,1), Fluid) ;
	nodeLayout.setNodeType (origin + Coordinates(2,3,2), Fluid) ;

}



void build_2T_cross (NodeLayout & nodeLayout)
{
	const unsigned width = 4 ;
	const unsigned height = 4 ;
	const unsigned depth = 8 ;

	Size size (width,height,depth) ;

	nodeLayout.resizeWithContent (size) ;

	for (unsigned z=0 ; z < depth ; z++)
		for (unsigned y=0 ; y < height ; y++)
			for (unsigned x=0 ; x < width ; x++)
			{
				nodeLayout.setNodeType (x,y,z, NodeBaseType::SOLID) ;
			}

	buildCross (nodeLayout, Coordinates(0,0,0) ) ;
	buildCross (nodeLayout, Coordinates(0,0,4) ) ;
}



void build_2T_fluid (NodeLayout & nodeLayout)
{
	const unsigned width = 4 ;
	const unsigned height = 4 ;
	const unsigned depth = 8 ;

	Size size (width,height,depth) ;

	nodeLayout.resizeWithContent (size) ;

	for (unsigned z=0 ; z < depth ; z++)
		for (unsigned y=0 ; y < height ; y++)
			for (unsigned x=0 ; x < width ; x++)
			{
				nodeLayout.setNodeType (x,y,z, NodeBaseType::FLUID) ;
			}
}



void build_2Tx2Tx2T_fluid (NodeLayout & nodeLayout)
{
	const unsigned width = 8 ;
	const unsigned height = 8 ;
	const unsigned depth = 8 ;

	Size size (width,height,depth) ;

	nodeLayout.resizeWithContent (size) ;

	for (unsigned z=0 ; z < depth ; z++)
		for (unsigned y=0 ; y < height ; y++)
			for (unsigned x=0 ; x < width ; x++)
			{
				nodeLayout.setNodeType (x,y,z, NodeBaseType::FLUID) ;
			}
}



TEST (Sparse, save)
{
	NodeLayout nodeLayout (Size(0,0,0)) ;

	PerformanceMeter pm ;
	
	pm.start() ;
	//build_2T_angle (nodeLayout) ;
	build_2T_cross (nodeLayout) ;
	//build_2T_fluid (nodeLayout) ;
	//build_2Tx2Tx2T_fluid (nodeLayout) ;
	pm.stop() ;
	logger << "NodeLayout filling: " << pm.generateSummary() << "\n" ;
	pm.clear() ;

	Size size = nodeLayout.getSize() ;
	const unsigned width  = size.getWidth() ;
	const unsigned height = size.getHeight() ;
	const unsigned depth  = size.getDepth() ;

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


	auto writer = vtkSmartPointer <WriterVtkImage>::New() ;
	writer->SetDataModeToBinary() ;

	stringstream ss ;
	ss << width << "x" << height << "x" << depth ;
	string fName ;

	pm.start() ;
	
	fName = testOutDirectory + "sparse_" + ss.str() ;
	logger << "Writing " << fName + ".vti" << "\n" ;
	writer->SetFileName ((fName + ".vti").c_str()) ;
	writer->SetCompressorTypeToZLib() ;
	writer->write (tiledLattice, CheckpointSettings()) ;

	pm.stop() ;
	logger << "Writing compressed: " << pm.generateSummary() << "\n" ;
	pm.clear() ;

	
	// Single cell in the middle.
	//vtkIdType pIds[] = {5,6,9,10,21,22,25,26} ;
	//unstructuredGrid->InsertNextCell (VTK_VOXEL, 8, pIds) ;


	auto unstructuredWriter = vtkSmartPointer <WriterVtkUnstructured>::New() ;

	pm.start() ;
	auto unstructuredGrid = 
		unstructuredWriter->buildUnstructuredGrid (tiledLattice, CheckpointSettings()) ;
	unstructuredWriter->addDataToGrid (unstructuredGrid, tiledLattice, CheckpointSettings()) ;
	pm.stop() ;
	logger << "Building vtkUnstructuredGrid: " << pm.generateSummary() << "\n" ;
	pm.clear() ;

	pm.start() ;
	std::stringstream origStream ;
	unstructuredWriter->registerOstream (origStream) ;
	unstructuredWriter->SetDataModeToAscii() ;
	unstructuredWriter->SetInput (unstructuredGrid) ;
	unstructuredWriter->Write() ;

	logger << "\n\nOriginal stream:\n\n" << origStream.str() << "\n" ;

	std::stringstream myStream ;
	unstructuredWriter->registerOstream (myStream) ;
	unstructuredWriter->write (tiledLattice, CheckpointSettings()) ;

	logger << "\n\nMy stream:\n\n" << myStream.str() << "\n" ;

	pm.stop() ;
	logger << "Writing unstructured: " << pm.generateSummary() << "\n" ;
	pm.clear() ;

	EXPECT_EQ (origStream.str(), myStream.str()) ;
}
