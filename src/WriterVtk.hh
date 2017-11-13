#ifndef WRITER_VTK_HH
#define WRITER_VTK_HH


#include <vtkIndent.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkStructuredData.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkCellType.h>


#include "VtkTypes.hpp"



namespace microflow
{



template <class VtkXmlWriterClass>
inline
void WriterVtkBase <VtkXmlWriterClass>::
registerOstream (std::ostream & os)
{
	this->Stream = & os ;
}



template <class VtkXmlWriterClass>
inline
vtkSmartPointer <vtkDoubleArray> WriterVtkBase <VtkXmlWriterClass>::
allocateArray1D (unsigned nElements)
{
	auto result = vtkSmartPointer <vtkDoubleArray>::New() ;

	result->SetNumberOfComponents (1) ;
	result->SetNumberOfTuples (nElements) ;

	return result ;
}



template <class VtkXmlWriterClass>
inline
vtkSmartPointer <vtkDoubleArray> WriterVtkBase <VtkXmlWriterClass>::
allocateArray3D (unsigned nElements)
{
	auto result = vtkSmartPointer <vtkDoubleArray>::New() ;

	result->SetNumberOfComponents (3) ;
	result->SetComponentName (0, "x") ;
	result->SetComponentName (1, "y") ;
	result->SetComponentName (2, "z") ;
	result->SetNumberOfTuples (nElements) ;

	return result ;
}



template <class VtkXmlWriterClass>
inline
void WriterVtkBase <VtkXmlWriterClass>::
fillArray (vtkAbstractArray * dataArray, double value)
{
	auto castedArray = vtkDataArray::SafeDownCast (dataArray) ;

	for (int componentIdx=0 ; 
			componentIdx < dataArray->GetNumberOfComponents() ; 
			componentIdx++)
	{
		castedArray->FillComponent (componentIdx, value) ;
	}
}



template <class VtkXmlWriterClass>
inline
void WriterVtkBase <VtkXmlWriterClass>::
SetDataMode (int mode) 
{ 
	BaseClass::SetDataMode (mode) ;
}



template <class VtkXmlWriterClass>
template <class TiledLattice, class Functor>
inline
void WriterVtkBase <VtkXmlWriterClass>::
forEachNode (TiledLattice const & tiledLattice, Functor functor)
{
	constexpr unsigned edge = TiledLattice::getNNodesPerTileEdge() ;

	const Size geometrySize = tiledLattice.getTileLayout().getNodeLayout().getSize() ;
	int geometryExtent[6] = {0,-1, 0,-1, 0,-1} ;
	geometryExtent[1] = geometrySize.getX() -1 ;
	geometryExtent[3] = geometrySize.getY() -1 ;
	geometryExtent[5] = geometrySize.getZ() -1 ;

	for (auto t = tiledLattice.getBeginOfTiles() ; t < tiledLattice.getEndOfTiles() ; t++)
	{
		auto tile = tiledLattice.getTile (t) ;
		Coordinates tileCorner = tile.getCornerPosition() ;

		for (unsigned tz=0 ; tz < edge ; tz++)
			for (unsigned ty=0 ; ty < edge ; ty++)
				for (unsigned tx=0 ; tx < edge ; tx++)
				{
					auto node = tile.getNode (tx,ty,tz) ;

					Coordinates globCoord = tileCorner + Coordinates (tx,ty,tz) ;
					int ijk[3] ;
					ijk[0] = globCoord.getX() ;
					ijk[1] = globCoord.getY() ;
					ijk[2] = globCoord.getZ() ;
					auto pointId = 
						vtkStructuredData::ComputePointIdForExtent (geometryExtent,ijk) ;

					functor (node, pointId) ;
				}
	}
}



template 
<
	class LatticeArrangement, 
	class DataType, 
	TileDataArrangement DataArrangement,
	class Settings
>
inline
void WriterVtkImage::
/*
	Based on vtkXMLStructuredDataWriter::ProcessRequest(...)
*/
write 
(
	TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
		& tiledLattice,
	Settings const & settings
)
{
	typedef TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement> 
		TiledLatticeType ;

	const Size geometrySize = tiledLattice.getTileLayout().getNodeLayout().getSize() ;
	double Origin[3] = {NAN,NAN,NAN} ;
	
	Origin [0] = settings.getGeometryOrigin().getX() ;
	Origin [1] = settings.getGeometryOrigin().getY() ;
	Origin [2] = settings.getGeometryOrigin().getZ() ;

	double Spacing[3] = {NAN,NAN,NAN} ;

	Spacing [0] = settings.getLatticeSpacingPhysical() ;
	Spacing [1] = settings.getLatticeSpacingPhysical() ;
	Spacing [2] = settings.getLatticeSpacingPhysical() ;

	int geometryExtent[6] = {0,-1, 0,-1, 0,-1} ;
	geometryExtent[1] = geometrySize.getX() -1 ;
	geometryExtent[3] = geometrySize.getY() -1 ;
	geometryExtent[5] = geometrySize.getZ() -1 ;
	const int nPoints = geometrySize.computeVolume() ;


	OpenFile() ;
	StartFile() ;


	/*
		Look at:
			vtkXMLStructuredDataWriter::WriteHeader()
			vtkXMLWriter::WritePrimaryElement(ostream &os, vtkIndent indent)
			vtkXMLImageDataWriter::WritePrimaryElementAttributes(ostream &os, vtkIndent indent)
	*/
  vtkIndent indent = vtkIndent().GetNextIndent();
  ostream& os = *(this->Stream);


  os << indent << "<" << this->GetDataSetName();
		this->WriteVectorAttribute("WholeExtent", 6, geometryExtent);
  	this->WriteVectorAttribute("Origin", 3, Origin);
  	this->WriteVectorAttribute("Spacing", 3, Spacing);
  os << ">\n";


	/*
		Look at:
			vtkXMLStructuredDataWriter::WriteAPiece()
			vtkXMLStructuredDataWriter::WriteInlineMode(vtkIndent indent)
			vtkXMLStructuredDataWriter::WriteInlinePiece(vtkIndent indent)
			vtkXMLWriter::WritePointDataInline(vtkPointData* pd, vtkIndent indent)
	*/
	os << indent << "<Piece";
		this->WriteVectorAttribute("Extent", 6, geometryExtent);
	os << ">\n";
	

	vtkIndent indent2 = indent.GetNextIndent();
  os << indent2 << "<PointData";
  os << ">\n";

	if (settings.shouldSaveNodes())
	{
		auto nodeBaseType          = vtkSmartPointer <vtkUnsignedCharArray>::New() ;
		auto nodePlacementModifier = vtkSmartPointer <vtkUnsignedCharArray>::New() ;

		nodeBaseType->SetName ("nodeBaseType") ;
		nodeBaseType->SetNumberOfComponents (1) ;
		nodeBaseType->SetNumberOfTuples (nPoints) ;
		nodePlacementModifier->SetName ("nodePlacementModifier") ;
		nodePlacementModifier->SetNumberOfComponents (1) ;
		nodePlacementModifier->SetNumberOfTuples (nPoints) ;

		fillArray (nodeBaseType, static_cast<double>(NodeBaseType::SOLID)) ;
		fillArray (nodePlacementModifier, static_cast<double>(PlacementModifier::NONE)) ;

		forEachNode (tiledLattice, 
			[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
					 vtkIdType pointId)
			{
				unsigned char baseType = static_cast<unsigned char>(node.nodeType().getBaseType()) ;
				nodeBaseType->SetValue (pointId, baseType) ;

				unsigned char placementModifier = 
					static_cast<unsigned char>(node.nodeType().getPlacementModifier()) ;
				nodePlacementModifier->SetValue (pointId, placementModifier) ;
			} 
		) ;
		this->WriteArrayInline (nodeBaseType, indent2.GetNextIndent()) ;
		this->WriteArrayInline (nodePlacementModifier, indent2.GetNextIndent()) ;
	}

	if (settings.shouldSaveVelocityLB() || settings.shouldSaveVelocityPhysical())
	{
		// WARNING - There is no method to remove component names, new array must
		//					 be created.
		auto dataArray = vtkSmartPointer <typename VtkTypes<DataType>::ArrayType>::New() ;

		dataArray->SetNumberOfComponents (3) ;
		dataArray->SetComponentName (0, "x") ;
		dataArray->SetComponentName (1, "y") ;
		dataArray->SetComponentName (2, "z") ;
		dataArray->SetNumberOfTuples (nPoints) ;

		if (settings.shouldSaveVelocityLB())
		{
			DataType uVec [3] ;

			dataArray->SetName ("boundaryVelocityLB") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						uVec [0] = node.uBoundary (Axis::X) ;
						uVec [1] = node.uBoundary (Axis::Y) ;
						uVec [2] = node.uBoundary (Axis::Z) ;

						dataArray->SetTupleValue (pointId, uVec) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;

			dataArray->SetName ("velocityLB") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						uVec [0] = node.u (Axis::X) ;
						uVec [1] = node.u (Axis::Y) ;
						uVec [2] = node.u (Axis::Z) ;

						dataArray->SetTupleValue (pointId, uVec) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;

			dataArray->SetName ("velocityT0LB") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						uVec [0] = node.uT0 (Axis::X) ;
						uVec [1] = node.uT0 (Axis::Y) ;
						uVec [2] = node.uT0 (Axis::Z) ;

						dataArray->SetTupleValue (pointId, uVec) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;
		}
			
		if (settings.shouldSaveVelocityPhysical())
		{
			DataType uVec [3] ;

			dataArray->SetName ("boundaryVelocity") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						uVec [0] = settings.transformVelocityLBToPhysical (node.uBoundary (Axis::X)) ;
						uVec [1] = settings.transformVelocityLBToPhysical (node.uBoundary (Axis::Y)) ;
						uVec [2] = settings.transformVelocityLBToPhysical (node.uBoundary (Axis::Z)) ;

						dataArray->SetTupleValue (pointId, uVec) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;

			dataArray->SetName ("velocity") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						uVec [0] = settings.transformVelocityLBToPhysical (node.u (Axis::X)) ;
						uVec [1] = settings.transformVelocityLBToPhysical (node.u (Axis::Y)) ;
						uVec [2] = settings.transformVelocityLBToPhysical (node.u (Axis::Z)) ;

						dataArray->SetTupleValue (pointId, uVec) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;
		}
	}


	if (settings.shouldSaveVolumetricMassDensityLB() || 
			settings.shouldSavePressurePhysical()        ||
			settings.shouldSaveMassFlowFractions())
	{
		DataType val ;

		auto dataArray = vtkSmartPointer <typename VtkTypes<DataType>::ArrayType>::New() ;
		dataArray->SetNumberOfComponents (1) ;
		dataArray->SetNumberOfTuples (nPoints) ;

		if (settings.shouldSaveVolumetricMassDensityLB())
		{
			dataArray->SetName ("boundaryRho") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						val = node.rhoBoundary() ;
						dataArray->SetTupleValue (pointId, & val) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;

			dataArray->SetName ("rhoLB") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						val = node.rho() ;
						dataArray->SetTupleValue (pointId, & val) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;

			dataArray->SetName ("rhoT0LB") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						val = node.rho0() ;
						dataArray->SetTupleValue (pointId, & val) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;
		}

		if (settings.shouldSavePressurePhysical())
		{
			dataArray->SetName ("pressure") ;
			fillArray (dataArray, NAN) ;
			forEachNode (tiledLattice, 
				[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
						 vtkIdType pointId)
					{
						val = node.rho() ;
						val = settings.transformVolumetricMassDensityLBToPressurePhysical (val) ;
						dataArray->SetTupleValue (pointId, & val) ;
					} 
				) ;
			this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;
		}

		if (settings.shouldSaveMassFlowFractions())
		{
			if (tiledLattice.isValidCopyIDNone())
			{
				THROW ("Undefined valid copy ID for tiledLattice") ;
			}

			for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
			{
				std::string fArrayName = buildFArrayName <LatticeArrangement> ("f", q) ;
				dataArray->SetName (fArrayName.c_str()) ;
				fillArray (dataArray, NAN) ;

				forEachNode (tiledLattice, 
					[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, 
							 vtkIdType pointId)
						{
							if (tiledLattice.isValidCopyIDF())
							{
								val = node.f (q) ;
							}
							else if (tiledLattice.isValidCopyIDFPost())
							{
								val = node.fPost (q) ;
							}

							dataArray->SetTupleValue (pointId, & val) ;
						} 
					) ;
				this->WriteArrayInline (dataArray, indent2.GetNextIndent()) ;
			}
		}
	}

	os << indent2 << "</PointData>\n";


  os << indent2 << "<CellData";
  os << ">\n";
  os << indent2 << "</CellData>\n";

	os << indent << "</Piece>\n";

	
	/*
		Look at:
			vtkXMLStructuredDataWriter::WriteFooter()
	 */
  os << indent << "</" << this->GetDataSetName() << ">\n";
    

	EndFile() ;
	CloseFile() ;
}



template 
<
	class LatticeArrangement, 
	class DataType, 
	TileDataArrangement DataArrangement,
	class Settings
>
inline
void WriterVtkUnstructured::
/*
	Based on vtkXMLUnstructuredDataWriter::ProcessRequest(...)
*/
write 
(
	TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
		& tiledLattice,
	Settings const & settings
)
{
	auto grid = buildUnstructuredGrid (tiledLattice, settings) ;

	const vtkIdType nCells  = grid->GetNumberOfCells() ;
	const vtkIdType nPoints = grid->GetNumberOfPoints() ;


	OpenFile() ;
	StartFile() ;


	/*
		Look at:
			vtkXMLUnstructuredDataWriter::WriteHeader()
			vtkXMLWriter::WritePrimaryElement(ostream &os, vtkIndent indent)
	*/
  vtkIndent indent = vtkIndent().GetNextIndent();
  ostream& os = *(this->Stream);


  os << indent << "<" << this->GetDataSetName();
  os << ">\n";


	/*
		Look at:
			vtkXMLUnstructuredDataWriter::WriteAPiece()
			vtkXMLUnstructuredGridWriter::WriteInlinePieceAttributes()
	*/
	vtkIndent indent2 = indent.GetNextIndent() ;
	os << indent2 << "<Piece";
		this->WriteScalarAttribute ("NumberOfPoints", nPoints) ;
		this->WriteScalarAttribute ("NumberOfCells" , nCells) ;
  os << ">\n";

	/*
		Look at:
			vtkXMLUnstructuredGridWriter::WriteInlinePiece(vtkIndent indent)
			vtkXMLUnstructuredDataWriter::WriteInlinePiece(vtkIndent indent)
	*/
	vtkIndent indent3 = indent2.GetNextIndent() ;

  os << indent3 << "<PointData";
  os << ">\n";

	if (settings.shouldSaveNodes())
	{
		auto nodeBaseType          = vtkSmartPointer <vtkUnsignedCharArray>::New() ;
		auto nodePlacementModifier = vtkSmartPointer <vtkUnsignedCharArray>::New() ;
		nodeBaseType->SetNumberOfComponents (1) ;
		nodePlacementModifier->SetNumberOfComponents (1) ;
		nodeBaseType->SetNumberOfTuples (nPoints) ;
		nodePlacementModifier->SetNumberOfTuples (nPoints) ;

		buildNodeArrays (nodeBaseType, nodePlacementModifier, tiledLattice) ;

		WriteArrayInline (nodeBaseType, indent3.GetNextIndent()) ;
		WriteArrayInline (nodePlacementModifier, indent3.GetNextIndent()) ;
	}

	if (settings.shouldSaveVelocityLB() || settings.shouldSaveVelocityPhysical())
	{
		auto dataArray = allocateArray3D (nPoints) ;

		if (settings.shouldSaveVelocityLB())
		{
			buildBoundaryVelocityLBArray (dataArray, tiledLattice) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;

			buildVelocityLBArray (dataArray, tiledLattice) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;

			buildVelocityT0LBArray (dataArray, tiledLattice) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;
		}

		if (settings.shouldSaveVelocityPhysical())
		{
			buildBoundaryVelocityPhysicalArray (dataArray, tiledLattice, settings) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;

			buildVelocityPhysicalArray (dataArray, tiledLattice, settings) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;
		}
	}

	if (settings.shouldSaveVolumetricMassDensityLB() ||
			settings.shouldSavePressurePhysical()        ||
			settings.shouldSaveMassFlowFractions())
	{
		auto dataArray = allocateArray1D (nPoints) ;

		if (settings.shouldSaveVolumetricMassDensityLB())
		{
			buildBoundaryRhoLBArray (dataArray, tiledLattice) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;

			buildRhoLBArray (dataArray, tiledLattice) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;

			buildRhoT0LBArray (dataArray, tiledLattice) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;
		}

		if (settings.shouldSavePressurePhysical())
		{
			buildPressureArray (dataArray, tiledLattice, settings) ;
			WriteArrayInline (dataArray, indent3.GetNextIndent()) ;
		}

		if (settings.shouldSaveMassFlowFractions())
		{
			if (tiledLattice.isValidCopyIDNone())
			{
				THROW ("Undefined valid copy ID for tiledLattice") ;
			}

			for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
			{
				buildFArray (dataArray, tiledLattice, q) ;
				WriteArrayInline (dataArray, indent3.GetNextIndent()) ;
			}
		}
	}
	
	os << indent3 << "</PointData>\n";

  os << indent3 << "<CellData";
  os << ">\n";
  os << indent3 << "</CellData>\n";

	this->WritePointsInline (grid->GetPoints(), indent3) ;
  this->WriteCellsInline  ("Cells", grid->GetCells(),
                         		grid->GetCellTypesArray(), 
                         		grid->GetFaces(),
                         		grid->GetFaceLocations(),
                         		indent3) ;

  os << indent2 << "</Piece>\n";

	/*
		Look at:
			vtkXMLUnstructuredDataWriter::WriteFooter()
	*/
	os << indent << "</" << this->GetDataSetName() << ">\n";


	EndFile() ;
	CloseFile() ;
}



template 
<
	class LatticeArrangement, 
	class DataType, 
	TileDataArrangement DataArrangement,
	class Settings
>
inline
vtkSmartPointer <vtkUnstructuredGrid> WriterVtkUnstructured::
buildUnstructuredGrid
(
	TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
		& tiledLattice,
	Settings const & settings
)
{
	typedef TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement> 
		TiledLatticeType ;

  auto grid = vtkSmartPointer <vtkUnstructuredGrid>::New();

	auto points = vtkSmartPointer <vtkPoints>::New() ;

	// First we add ALL nodes (points) from tiles to simpify index computation.
	// The number of unused points (SOLID nodes) should not be too large due to tiling.
	// TODO: Count number of added solid nodes. (About 30% even for large microchannels).
	tiledLattice.forEachNode // Remember to keep XYZ order of nodes.
	(
		[&] (typename TiledLatticeType::TileType::DefaultNodeType & node,
				 Coordinates & globalNodeCoordinates)
		{
			double x = 
				((long double)globalNodeCoordinates.getX()) * 
				settings.getLatticeSpacingPhysical() +
				settings.getGeometryOrigin().getX() ;

			double y = 
				((long double)globalNodeCoordinates.getY()) * 
				settings.getLatticeSpacingPhysical() +
				settings.getGeometryOrigin().getY() ;

			double z = 
				((long double)globalNodeCoordinates.getZ()) *
				settings.getLatticeSpacingPhysical() +
				settings.getGeometryOrigin().getZ() ;

			auto pId = points->InsertNextPoint (x,y,z) ;
		}
	) ;

  grid->SetPoints (points) ;

	// Next we add only these cells, which have ALL corners not solid.
	tiledLattice.forEachTile
	(
		[&] (typename TiledLatticeType::TileType & tile)
		{
			addCellsFromTile (tile, grid) ;

			//	Cells between tiles.
			//	Tiles are ordered by X,Y,Z coordinates, thus we check only these tiles,
			//	which were already checked.
			//
			addCellsFromPlaneB (tile, grid) ;
			addCellsFromPlaneW (tile, grid) ;
			addCellsFromPlaneS (tile, grid) ;

			addCellsFromEdgeWB (tile,grid) ;
			addCellsFromEdgeSB (tile,grid) ;
			addCellsFromEdgeSW (tile,grid) ;
			
			addCellFromCorner (tile,grid) ;
		}
	) ;

  return grid ;	
}



template 
<
	class LatticeArrangement, 
	class DataType, 
	TileDataArrangement DataArrangement,
	class Settings
>
inline
void WriterVtkUnstructured::
addDataToGrid
(
	vtkSmartPointer <vtkUnstructuredGrid> & grid,
	TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
		& tiledLattice,
	const Settings & settings
)
{
	typedef TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement> 
		TiledLatticeType ;

	const unsigned nPoints = tiledLattice.getNOfTiles() * 
													 TiledLatticeType::TileType::getNNodesPerTile() ;

	if (settings.shouldSaveNodes())
	{
		auto nodeBaseType          = vtkSmartPointer <vtkUnsignedCharArray>::New() ;
		auto nodePlacementModifier = vtkSmartPointer <vtkUnsignedCharArray>::New() ;
		nodeBaseType->SetNumberOfComponents (1) ;
		nodePlacementModifier->SetNumberOfComponents (1) ;
		nodeBaseType->SetNumberOfTuples (nPoints) ;
		nodePlacementModifier->SetNumberOfTuples (nPoints) ;

		buildNodeArrays (nodeBaseType, nodePlacementModifier, tiledLattice) ;

		grid->GetPointData()->AddArray (nodeBaseType) ;
		grid->GetPointData()->AddArray (nodePlacementModifier) ;
	}

	if (settings.shouldSaveVelocityLB())
	{
		auto boundaryVelocityLB = allocateArray3D (nPoints) ;
		buildBoundaryVelocityLBArray (boundaryVelocityLB, tiledLattice) ;
		grid->GetPointData()->AddArray (boundaryVelocityLB) ;

		auto velocityLB = allocateArray3D (nPoints) ;
		buildVelocityLBArray (velocityLB, tiledLattice) ;
		grid->GetPointData()->AddArray (velocityLB) ;

		auto velocityT0LB = allocateArray3D (nPoints) ;
		buildVelocityT0LBArray (velocityT0LB, tiledLattice) ;
		grid->GetPointData()->AddArray (velocityT0LB) ;
	}

	if (settings.shouldSaveVelocityPhysical())
	{
		auto boundaryVelocity = allocateArray3D (nPoints) ;
		buildBoundaryVelocityPhysicalArray (boundaryVelocity, tiledLattice, settings) ;
		grid->GetPointData()->AddArray (boundaryVelocity) ;

		auto velocity = allocateArray3D (nPoints) ;
		buildVelocityPhysicalArray (velocity, tiledLattice, settings) ;
		grid->GetPointData()->AddArray (velocity) ;
	}

	if (settings.shouldSaveVolumetricMassDensityLB())
	{
		auto boundaryRho = allocateArray1D (nPoints) ;
		buildBoundaryRhoLBArray (boundaryRho, tiledLattice) ;
		grid->GetPointData()->AddArray (boundaryRho) ;

		auto rhoLB = allocateArray1D (nPoints) ;
		buildRhoLBArray (rhoLB, tiledLattice) ;
		grid->GetPointData()->AddArray (rhoLB) ;

		auto rhoT0LB = allocateArray1D (nPoints) ;
		buildRhoT0LBArray (rhoT0LB, tiledLattice) ;
		grid->GetPointData()->AddArray (rhoT0LB) ;
	}

	if (settings.shouldSavePressurePhysical())
	{
		auto pressure = allocateArray1D (nPoints) ;
		buildPressureArray (pressure, tiledLattice, settings) ;
		grid->GetPointData()->AddArray (pressure) ;
	}

	if (settings.shouldSaveMassFlowFractions())
	{
		if (tiledLattice.isValidCopyIDNone())
		{
			THROW ("Undefined valid copy ID for tiledLattice") ;
		}

		for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
		{
			auto fArray = allocateArray1D (nPoints) ;
			buildFArray (fArray, tiledLattice, q) ;
			grid->GetPointData()->AddArray (fArray) ;
		}
	}
}



template <class Tile>
inline
void WriterVtkUnstructured::
addCellsFromTile
(
	Tile & tile,
	vtkSmartPointer <vtkUnstructuredGrid> & grid
)
{
	/*
		Nodes in tile are ordered using XYZ mapper.
	*/
	const unsigned edge = tile.getNNodesPerEdge() ;

	for (unsigned cellZ=0 ; cellZ < (edge-1) ; cellZ++)
		for (unsigned cellY=0 ; cellY < (edge-1) ; cellY++)
			for (unsigned cellX=0 ; cellX < (edge-1) ; cellX++)
			{
				if (
						not tile.getNode (cellX  , cellY  , cellZ  ).nodeType().isSolid()  &&
						not tile.getNode (cellX+1, cellY  , cellZ  ).nodeType().isSolid()  &&
						not tile.getNode (cellX  , cellY+1, cellZ  ).nodeType().isSolid()  &&
						not tile.getNode (cellX+1, cellY+1, cellZ  ).nodeType().isSolid()  &&
						not tile.getNode (cellX  , cellY  , cellZ+1).nodeType().isSolid()  &&
						not tile.getNode (cellX+1, cellY  , cellZ+1).nodeType().isSolid()  &&
						not tile.getNode (cellX  , cellY+1, cellZ+1).nodeType().isSolid()  &&
						not tile.getNode (cellX+1, cellY+1, cellZ+1).nodeType().isSolid()  
					 )
				{
					vtkIdType pIds [8] ;
		
					pIds [0] = computeGlobalIndex (tile, cellX  , cellY  , cellZ  ) ;
					pIds [1] = computeGlobalIndex (tile, cellX+1, cellY  , cellZ  ) ;
					pIds [2] = computeGlobalIndex (tile, cellX  , cellY+1, cellZ  ) ;
					pIds [3] = computeGlobalIndex (tile, cellX+1, cellY+1, cellZ  ) ;
					pIds [4] = computeGlobalIndex (tile, cellX  , cellY  , cellZ+1) ;
					pIds [5] = computeGlobalIndex (tile, cellX+1, cellY  , cellZ+1) ;
					pIds [6] = computeGlobalIndex (tile, cellX  , cellY+1, cellZ+1) ;
					pIds [7] = computeGlobalIndex (tile, cellX+1, cellY+1, cellZ+1) ;
					
					grid->InsertNextCell (VTK_VOXEL, 8, pIds) ;
				}
			}
}



template <class Tile>
inline
void WriterVtkUnstructured::
addCellsFromPlaneB
(
	Tile & tile,
	vtkSmartPointer <vtkUnstructuredGrid> & grid
)
{
	auto neighborTile = tile.getNeighbor (B) ;
	if (not neighborTile.isEmpty())
	{
		const unsigned edge = tile.getNNodesPerEdge() ;

		for (unsigned cellY=0 ; cellY < (edge-1) ; cellY++)
			for (unsigned cellX=0 ; cellX < (edge-1) ; cellX++)
			{
				if (
						not tile.getNode (cellX  , cellY  , 0).nodeType().isSolid()  &&
						not tile.getNode (cellX+1, cellY  , 0).nodeType().isSolid()  &&
						not tile.getNode (cellX  , cellY+1, 0).nodeType().isSolid()  &&
						not tile.getNode (cellX+1, cellY+1, 0).nodeType().isSolid()  &&

						not neighborTile.getNode (cellX  , cellY  , edge-1).nodeType().isSolid()  &&
						not neighborTile.getNode (cellX+1, cellY  , edge-1).nodeType().isSolid()  &&
						not neighborTile.getNode (cellX  , cellY+1, edge-1).nodeType().isSolid()  &&
						not neighborTile.getNode (cellX+1, cellY+1, edge-1).nodeType().isSolid()
					 )
				{
					vtkIdType pIds [8] ;

					pIds [0] = computeGlobalIndex (neighborTile, cellX  , cellY  , edge-1) ;
					pIds [1] = computeGlobalIndex (neighborTile, cellX+1, cellY  , edge-1) ;
					pIds [2] = computeGlobalIndex (neighborTile, cellX  , cellY+1, edge-1) ;
					pIds [3] = computeGlobalIndex (neighborTile, cellX+1, cellY+1, edge-1) ;
					pIds [4] = computeGlobalIndex (tile        , cellX  , cellY  , 0     ) ;
					pIds [5] = computeGlobalIndex (tile        , cellX+1, cellY  , 0     ) ;
					pIds [6] = computeGlobalIndex (tile        , cellX  , cellY+1, 0     ) ;
					pIds [7] = computeGlobalIndex (tile        , cellX+1, cellY+1, 0     ) ;

					grid->InsertNextCell (VTK_VOXEL, 8, pIds) ;
				}
			}
	}
}



template <class Tile>
inline
void WriterVtkUnstructured::
addCellsFromPlaneW
(
	Tile & tile,
	vtkSmartPointer <vtkUnstructuredGrid> & grid
)
{
	auto neighborTile = tile.getNeighbor (W) ;
	if (not neighborTile.isEmpty())
	{
		const unsigned edge = tile.getNNodesPerEdge() ;

		for (unsigned cellZ=0 ; cellZ < (edge-1) ; cellZ++)
			for (unsigned cellY=0 ; cellY < (edge-1) ; cellY++)
			{
				if (
						not tile.getNode (0, cellY  , cellZ  ).nodeType().isSolid()  &&
						not tile.getNode (0, cellY  , cellZ+1).nodeType().isSolid()  &&
						not tile.getNode (0, cellY+1, cellZ  ).nodeType().isSolid()  &&
						not tile.getNode (0, cellY+1, cellZ+1).nodeType().isSolid()  &&

						not neighborTile.getNode (edge-1, cellY  , cellZ  ).nodeType().isSolid()  &&
						not neighborTile.getNode (edge-1, cellY  , cellZ+1).nodeType().isSolid()  &&
						not neighborTile.getNode (edge-1, cellY+1, cellZ  ).nodeType().isSolid()  &&
						not neighborTile.getNode (edge-1, cellY+1, cellZ+1).nodeType().isSolid()
					 )
				{
					vtkIdType pIds [8] ;

					pIds [0] = computeGlobalIndex (neighborTile, edge-1, cellY  , cellZ  ) ;
					pIds [1] = computeGlobalIndex (tile        , 0     , cellY  , cellZ  ) ;
					pIds [2] = computeGlobalIndex (neighborTile, edge-1, cellY+1, cellZ  ) ;
					pIds [3] = computeGlobalIndex (tile        , 0     , cellY+1, cellZ  ) ;
					pIds [4] = computeGlobalIndex (neighborTile, edge-1, cellY  , cellZ+1) ;
					pIds [5] = computeGlobalIndex (tile        , 0     , cellY  , cellZ+1) ;
					pIds [6] = computeGlobalIndex (neighborTile, edge-1, cellY+1, cellZ+1) ;
					pIds [7] = computeGlobalIndex (tile        , 0     , cellY+1, cellZ+1) ;

					grid->InsertNextCell (VTK_VOXEL, 8, pIds) ;
				}
			}
	}
}



template <class Tile>
inline
void WriterVtkUnstructured::
addCellsFromPlaneS
(
	Tile & tile,
	vtkSmartPointer <vtkUnstructuredGrid> & grid
)
{
	auto neighborTile = tile.getNeighbor (S) ;
	if (not neighborTile.isEmpty())
	{
		const unsigned edge = tile.getNNodesPerEdge() ;

		for (unsigned cellZ=0 ; cellZ < (edge-1) ; cellZ++)
			for (unsigned cellX=0 ; cellX < (edge-1) ; cellX++)
			{
				if (
						not tile.getNode (cellX  , 0, cellZ  ).nodeType().isSolid()  &&
						not tile.getNode (cellX  , 0, cellZ+1).nodeType().isSolid()  &&
						not tile.getNode (cellX+1, 0, cellZ  ).nodeType().isSolid()  &&
						not tile.getNode (cellX+1, 0, cellZ+1).nodeType().isSolid()  &&

						not neighborTile.getNode (cellX  , edge-1, cellZ  ).nodeType().isSolid()  &&
						not neighborTile.getNode (cellX  , edge-1, cellZ+1).nodeType().isSolid()  &&
						not neighborTile.getNode (cellX+1, edge-1, cellZ  ).nodeType().isSolid()  &&
						not neighborTile.getNode (cellX+1, edge-1, cellZ+1).nodeType().isSolid()
					 )
				{
					vtkIdType pIds [8] ;

					pIds [0] = computeGlobalIndex (neighborTile, cellX  , edge-1, cellZ  ) ;
					pIds [1] = computeGlobalIndex (neighborTile, cellX+1, edge-1, cellZ  ) ;
					pIds [2] = computeGlobalIndex (tile        , cellX  , 0     , cellZ  ) ;
					pIds [3] = computeGlobalIndex (tile        , cellX+1, 0     , cellZ  ) ;
					pIds [4] = computeGlobalIndex (neighborTile, cellX  , edge-1, cellZ+1) ;
					pIds [5] = computeGlobalIndex (neighborTile, cellX+1, edge-1, cellZ+1) ;
					pIds [6] = computeGlobalIndex (tile        , cellX  , 0     , cellZ+1) ;
					pIds [7] = computeGlobalIndex (tile        , cellX+1, 0     , cellZ+1) ;

					grid->InsertNextCell (VTK_VOXEL, 8, pIds) ;
				}
			}
	}
}



template <class Tile>
inline
void WriterVtkUnstructured::
addCellsFromEdgeWB
(
	Tile & tile,
	vtkSmartPointer <vtkUnstructuredGrid> & grid
)
{
	auto neighborTileB = tile.getNeighbor (B) ;
	auto neighborTileW = tile.getNeighbor (W) ;
	auto neighborTileWB = tile.getNeighbor (WB) ;

	if (
			neighborTileB.isEmpty() ||
			neighborTileW.isEmpty() ||
			neighborTileWB.isEmpty()
		 )
	{
		return ;
	}

	const unsigned edge = tile.getNNodesPerEdge() ;

	for (unsigned cellY=0 ; cellY < (edge-1) ; cellY++)
	{
		if (
				not tile.getNode (0, cellY  , 0).nodeType().isSolid()  &&
				not tile.getNode (0, cellY+1, 0).nodeType().isSolid()  &&

				not neighborTileW.getNode (edge-1, cellY  , 0).nodeType().isSolid()  &&
				not neighborTileW.getNode (edge-1, cellY+1, 0).nodeType().isSolid()  &&

				not neighborTileB.getNode (0, cellY  , edge-1).nodeType().isSolid()  &&
				not neighborTileB.getNode (0, cellY+1, edge-1).nodeType().isSolid()  &&

				not neighborTileWB.getNode (edge-1, cellY  , edge-1).nodeType().isSolid()  &&
				not neighborTileWB.getNode (edge-1, cellY+1, edge-1).nodeType().isSolid()
			 )
		{
			vtkIdType pIds [8] ;

			pIds [0] = computeGlobalIndex (neighborTileWB, edge-1, cellY, edge-1) ;
			pIds [1] = computeGlobalIndex (neighborTileB , 0     , cellY, edge-1) ;

			pIds [2] = computeGlobalIndex (neighborTileWB, edge-1, cellY+1, edge-1) ;
			pIds [3] = computeGlobalIndex (neighborTileB , 0     , cellY+1, edge-1) ;
			
			pIds [4] = computeGlobalIndex (neighborTileW, edge-1, cellY, 0) ;
			pIds [5] = computeGlobalIndex (tile         , 0     , cellY, 0) ;

			pIds [6] = computeGlobalIndex (neighborTileW, edge-1, cellY+1, 0) ;
			pIds [7] = computeGlobalIndex (tile         , 0     , cellY+1, 0) ;

			grid->InsertNextCell (VTK_VOXEL, 8, pIds) ;
		}
	}
}



template <class Tile>
inline
void WriterVtkUnstructured::
addCellsFromEdgeSB
(
	Tile & tile,
	vtkSmartPointer <vtkUnstructuredGrid> & grid
)
{
	auto neighborTileB = tile.getNeighbor (B) ;
	auto neighborTileS = tile.getNeighbor (S) ;
	auto neighborTileSB = tile.getNeighbor (SB) ;

	if (
			neighborTileB.isEmpty() ||
			neighborTileS.isEmpty() ||
			neighborTileSB.isEmpty()
		 )
	{
		return ;
	}

	const unsigned edge = tile.getNNodesPerEdge() ;

	for (unsigned cellX=0 ; cellX < (edge-1) ; cellX++)
	{
		if (
				not tile.getNode (cellX  , 0, 0).nodeType().isSolid()  &&
				not tile.getNode (cellX+1, 0, 0).nodeType().isSolid()  &&

				not neighborTileS.getNode (cellX  , edge-1, 0).nodeType().isSolid()  &&
				not neighborTileS.getNode (cellX+1, edge-1, 0).nodeType().isSolid()  &&

				not neighborTileB.getNode (cellX  , 0, edge-1).nodeType().isSolid()  &&
				not neighborTileB.getNode (cellX+1, 0, edge-1).nodeType().isSolid()  &&

				not neighborTileSB.getNode (cellX  , edge-1, edge-1).nodeType().isSolid()  &&
				not neighborTileSB.getNode (cellX+1, edge-1, edge-1).nodeType().isSolid()
			 )
		{
			vtkIdType pIds [8] ;

			pIds [0] = computeGlobalIndex (neighborTileSB, cellX  , edge-1, edge-1) ;
			pIds [1] = computeGlobalIndex (neighborTileSB, cellX+1, edge-1, edge-1) ;

			pIds [2] = computeGlobalIndex (neighborTileB , cellX  , 0     , edge-1) ;
			pIds [3] = computeGlobalIndex (neighborTileB , cellX+1, 0     , edge-1) ;
			
			pIds [4] = computeGlobalIndex (neighborTileS, cellX  , edge-1, 0) ;
			pIds [5] = computeGlobalIndex (neighborTileS, cellX+1, edge-1, 0) ;

			pIds [6] = computeGlobalIndex (tile         , cellX  , 0     , 0) ;
			pIds [7] = computeGlobalIndex (tile         , cellX+1, 0     , 0) ;

			grid->InsertNextCell (VTK_VOXEL, 8, pIds) ;
		}
	}
}



template <class Tile>
inline
void WriterVtkUnstructured::
addCellsFromEdgeSW
(
	Tile & tile,
	vtkSmartPointer <vtkUnstructuredGrid> & grid
)
{
	auto neighborTileW = tile.getNeighbor (W) ;
	auto neighborTileS = tile.getNeighbor (S) ;
	auto neighborTileSW = tile.getNeighbor (SW) ;

	if (
			neighborTileW.isEmpty() ||
			neighborTileS.isEmpty() ||
			neighborTileSW.isEmpty()
		 )
	{
		return ;
	}

	const unsigned edge = tile.getNNodesPerEdge() ;

	for (unsigned cellZ=0 ; cellZ < (edge-1) ; cellZ++)
	{
		if (
				not tile.getNode (0, 0, cellZ  ).nodeType().isSolid()  &&
				not tile.getNode (0, 0, cellZ+1).nodeType().isSolid()  &&

				not neighborTileS.getNode (0, edge-1, cellZ  ).nodeType().isSolid()  &&
				not neighborTileS.getNode (0, edge-1, cellZ+1).nodeType().isSolid()  &&

				not neighborTileW.getNode (edge-1, 0, cellZ  ).nodeType().isSolid()  &&
				not neighborTileW.getNode (edge-1, 0, cellZ+1).nodeType().isSolid()  &&

				not neighborTileSW.getNode (edge-1, edge-1, cellZ  ).nodeType().isSolid()  &&
				not neighborTileSW.getNode (edge-1, edge-1, cellZ+1).nodeType().isSolid()
			 )
		{
			vtkIdType pIds [8] ;

			pIds [0] = computeGlobalIndex (neighborTileSW, edge-1, edge-1, cellZ  ) ;
			pIds [1] = computeGlobalIndex (neighborTileS , 0     , edge-1, cellZ  ) ;
			pIds [2] = computeGlobalIndex (neighborTileW , edge-1, 0     , cellZ  ) ;
			pIds [3] = computeGlobalIndex (tile          , 0     , 0     , cellZ  ) ;
			pIds [4] = computeGlobalIndex (neighborTileSW, edge-1, edge-1, cellZ+1) ;
			pIds [5] = computeGlobalIndex (neighborTileS , 0     , edge-1, cellZ+1) ;
			pIds [6] = computeGlobalIndex (neighborTileW , edge-1, 0     , cellZ+1) ;
			pIds [7] = computeGlobalIndex (tile          , 0     , 0     , cellZ+1) ;

			grid->InsertNextCell (VTK_VOXEL, 8, pIds) ;
		}
	}
}



template <class Tile>
inline
void WriterVtkUnstructured::
addCellFromCorner
(
	Tile & tile,
	vtkSmartPointer <vtkUnstructuredGrid> & grid
)
{
	auto neighborTileB = tile.getNeighbor (B) ;
	auto neighborTileS = tile.getNeighbor (S) ;
	auto neighborTileW = tile.getNeighbor (W) ;

	auto neighborTileSW = tile.getNeighbor (SW) ;
	auto neighborTileSB = tile.getNeighbor (SB) ;
	auto neighborTileWB = tile.getNeighbor (WB) ;

	auto neighborTileSWB = tile.getNeighbor (SWB) ;

	if (
			neighborTileB.isEmpty() ||
			neighborTileS.isEmpty() ||
			neighborTileW.isEmpty() ||

			neighborTileSW.isEmpty() ||
			neighborTileSB.isEmpty() ||
			neighborTileWB.isEmpty() ||

			neighborTileSWB.isEmpty()
		 )
	{
		return ;
	}

	const unsigned edge = tile.getNNodesPerEdge() ;

	if (
			not tile.getNode (0, 0, 0).nodeType().isSolid()  &&

			not neighborTileW.getNode (edge-1, 0, 0).nodeType().isSolid()  &&
			not neighborTileS.getNode (0, edge-1, 0).nodeType().isSolid()  &&
			not neighborTileB.getNode (0, 0, edge-1).nodeType().isSolid()  &&

			not neighborTileSW.getNode (edge-1, edge-1, 0).nodeType().isSolid()  &&
			not neighborTileSB.getNode (0, edge-1, edge-1).nodeType().isSolid()  &&
			not neighborTileWB.getNode (edge-1, 0, edge-1).nodeType().isSolid()  &&

			not neighborTileSWB.getNode (edge-1, edge-1, edge-1).nodeType().isSolid()
		 )
	{
		vtkIdType pIds [8] ;

		pIds [0] = computeGlobalIndex (neighborTileSWB, edge-1, edge-1, edge-1) ;
		pIds [1] = computeGlobalIndex (neighborTileSB , 0     , edge-1, edge-1) ;
		pIds [2] = computeGlobalIndex (neighborTileWB , edge-1, 0     , edge-1) ;
		pIds [3] = computeGlobalIndex (neighborTileB  , 0     , 0     , edge-1) ;
		pIds [4] = computeGlobalIndex (neighborTileSW , edge-1, edge-1, 0     ) ;
		pIds [5] = computeGlobalIndex (neighborTileS  , 0     , edge-1, 0     ) ;
		pIds [6] = computeGlobalIndex (neighborTileW  , edge-1, 0     , 0     ) ;
		pIds [7] = computeGlobalIndex (tile, 0,0,0) ;

		grid->InsertNextCell (VTK_VOXEL, 8, pIds) ;
	}
}



template <class Tile>
inline
unsigned WriterVtkUnstructured::
computeGlobalIndex (Tile & tile,
										unsigned nodeInTileX, 
										unsigned nodeInTileY,
										unsigned nodeInTileZ)
{
	return tile.computeNodeIndex (nodeInTileX, nodeInTileY, nodeInTileZ,
																tile.getCurrentTileIndex()) ;
}



template <class TiledLattice>
inline
void WriterVtkUnstructured::
buildNodeArrays
(
	vtkSmartPointer <vtkUnsignedCharArray> & nodeBaseType,
	vtkSmartPointer <vtkUnsignedCharArray> & nodePlacementModifier,
	TiledLattice & tiledLattice
)
{
	nodeBaseType->SetName ("nodeBaseType") ;
	nodePlacementModifier->SetName ("nodePlacementModifier") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode 
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node,
				 Coordinates & globCoord)
		{
			unsigned char baseType = static_cast<unsigned char>(node.nodeType().getBaseType()) ;
			nodeBaseType->SetValue (pointId, baseType) ;
			unsigned char placementModifier = 
				static_cast<unsigned char>(node.nodeType().getPlacementModifier()) ;
			nodePlacementModifier->SetValue (pointId, placementModifier) ;

			pointId ++ ;
		}
	) ;
}



/*
	TODO: Replace below methods with something similar to ReaderVtk interface.
*/
template <class TiledLattice>
inline
void WriterVtkUnstructured::
buildBoundaryVelocityLBArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice
)
{
	dataArray->SetName ("boundaryVelocityLB") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType uVec [3] ;

				uVec [0] = node.uBoundary (Axis::X) ;
				uVec [1] = node.uBoundary (Axis::Y) ;
				uVec [2] = node.uBoundary (Axis::Z) ;

				dataArray->SetTupleValue (pointId, uVec) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice>
inline
void WriterVtkUnstructured::
buildVelocityLBArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice
)
{
	dataArray->SetName ("velocityLB") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType uVec [3] ;

				uVec [0] = node.u (Axis::X) ;
				uVec [1] = node.u (Axis::Y) ;
				uVec [2] = node.u (Axis::Z) ;

				dataArray->SetTupleValue (pointId, uVec) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice>
inline
void WriterVtkUnstructured::
buildVelocityT0LBArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice
)
{
	dataArray->SetName ("velocityT0LB") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType uVec [3] ;

				uVec [0] = node.uT0 (Axis::X) ;
				uVec [1] = node.uT0 (Axis::Y) ;
				uVec [2] = node.uT0 (Axis::Z) ;

				dataArray->SetTupleValue (pointId, uVec) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice, class Settings>
inline
void WriterVtkUnstructured::
buildBoundaryVelocityPhysicalArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice,
	const Settings & settings
)
{
	dataArray->SetName ("boundaryVelocity") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType uVec [3] ;

				uVec [0] = settings.transformVelocityLBToPhysical (node.uBoundary (Axis::X)) ;
				uVec [1] = settings.transformVelocityLBToPhysical (node.uBoundary (Axis::Y)) ;
				uVec [2] = settings.transformVelocityLBToPhysical (node.uBoundary (Axis::Z)) ;

				dataArray->SetTupleValue (pointId, uVec) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice, class Settings>
inline
void WriterVtkUnstructured::
buildVelocityPhysicalArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice,
	const Settings & settings
)
{
	dataArray->SetName ("velocity") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType uVec [3] ;

				uVec [0] = settings.transformVelocityLBToPhysical (node.u (Axis::X)) ;
				uVec [1] = settings.transformVelocityLBToPhysical (node.u (Axis::Y)) ;
				uVec [2] = settings.transformVelocityLBToPhysical (node.u (Axis::Z)) ;

				dataArray->SetTupleValue (pointId, uVec) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice>
inline
void WriterVtkUnstructured::
buildBoundaryRhoLBArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice
)
{
	dataArray->SetName ("boundaryRho") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType val = node.rhoBoundary() ;

				dataArray->SetTupleValue (pointId, & val) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice>
inline
void WriterVtkUnstructured::
buildRhoLBArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice
)
{
	dataArray->SetName ("rhoLB") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType val = node.rho() ;

				dataArray->SetTupleValue (pointId, & val) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice>
inline
void WriterVtkUnstructured::
buildRhoT0LBArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice
)
{
	dataArray->SetName ("rhoT0LB") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType val = node.rho0() ;

				dataArray->SetTupleValue (pointId, & val) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice, class Settings>
inline
void WriterVtkUnstructured::
buildPressureArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice,
	const Settings & settings
)
{
	dataArray->SetName ("pressure") ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType val = node.rho() ;
				val = settings.transformVolumetricMassDensityLBToPressurePhysical (val) ;

				dataArray->SetTupleValue (pointId, & val) ;

				pointId ++ ;
			} 
		) ;
}



template <class TiledLattice>
inline
void WriterVtkUnstructured::
buildFArray
(
	vtkSmartPointer <vtkDoubleArray> & dataArray,
	TiledLattice & tiledLattice,
	Direction::DirectionIndex q
)
{
	std::string fArrayName = 
		buildFArrayName <typename TiledLattice::LatticeArrangementType> ("f", q) ;
	dataArray->SetName (fArrayName.c_str()) ;

	vtkIdType pointId = 0 ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node, 
				 Coordinates & globCoord)
			{
				typename TiledLattice::DataTypeType val ;

				if (tiledLattice.isValidCopyIDF())
				{
					val = node.f (q) ;
				}
				else if (tiledLattice.isValidCopyIDFPost())
				{
					val = node.fPost (q) ;
				}

				dataArray->SetTupleValue (pointId, & val) ;

				pointId ++ ;
			} 
		) ;
}



}



#endif
