#ifndef READER_VTK_HH
#define READER_VTK_HH



#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkType.h>


namespace microflow
{



template <class VtkXmlReaderClass>
inline	
void ReaderVtkBase <VtkXmlReaderClass>::
registerIstream (std::istream & is)
{
	this->Stream = & is ;
}



template <class VtkXmlReaderClass>
template 
<
	class LatticeArrangement, 
	class DataType, 
	TileDataArrangement DataArrangement
>
inline
void ReaderVtkBase <VtkXmlReaderClass>::
read 
(
	TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
		& tiledLattice
)
{
	typedef TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement> 
						TLattice ;

	readSkeleton (tiledLattice,
		
		[&] (TLattice & tiledLattice)
		{
			this->readTiledLattice (tiledLattice) ;
		}
	) ;
}



template <class VtkXmlReaderClass>
inline	
vtkXMLDataElement * ReaderVtkBase <VtkXmlReaderClass>::
findArrayDataElement (vtkXMLDataElement * dataElements, std::string name)
{
	for (int i=0 ; i < dataElements->GetNumberOfNestedElements() ; ++i)
	{
		vtkXMLDataElement * element = dataElements->GetNestedElement (i) ;
		const char * eName = element->GetAttribute ("Name") ;

		if (name == eName)
		{
			return element ;
		}
	}

	return NULL ;
}



template <class VtkXmlReaderClass>
inline	
vtkAbstractArray * ReaderVtkBase <VtkXmlReaderClass>::
readDataArray (vtkXMLDataElement * dataElements, std::string name, unsigned nValues)
{
	vtkXMLDataElement* element = findArrayDataElement (dataElements, name) ;

	if (NULL == element)
	{
		THROW ("Can not find \"" + name + "\" in vtk file") ;
	}

	if( strcmp(element->GetName(), "DataArray") != 0  && 
			strcmp(element->GetName(), "Array") != 0 )
	{
		THROW ("Wrong type of \"" + name + "\" in vtk file") ;
	}

	vtkAbstractArray * arrayPtr = this->CreateArray (element) ;
	arrayPtr->SetNumberOfTuples (nValues);

	/*
		 Look at:
		 vtkXMLDataReader::ReadXMLData()
	 */
	if (!this->ReadArrayValues (element, 0, arrayPtr, 0, 
				nValues * arrayPtr->GetNumberOfComponents()))
	{
		THROW ("Can not read array " + name) ;
	}

	return arrayPtr ;
}



template <class VtkXmlReaderClass>
template <class Argument, class Reader>
inline
void ReaderVtkBase <VtkXmlReaderClass>::
/*
	Based on
		vtkXMLReader::RequestData(vtkInformation *vtkNotUsed(request),
                              vtkInformationVector **vtkNotUsed(inputVector),
                              vtkInformationVector *outputVector)
*/
readSkeleton (Argument & argument, Reader reader)
{
	//FIXME: Does not work, probably information must be configured in some way.
	//auto information = vtkSmartPointer <vtkInformation>::New() ;
	//this->CurrentOutputInformation = information ;

	this->ReadXMLInformation() ; // calls ReadPrimaryElement()


	/*
		Look at:
			vtkXMLReader::RequestData
			vtkXMLReader::ReadXMLData()
			vtkXMLDataReader::ReadXMLData()
			vtkXMLStructuredDataReader::ReadXMLData()
	 */
	if(!this->OpenVTKFile())
	{
		THROW ("Can not open VTK file") ;
	}

	if(!this->XMLParser)
	{
		THROW ("ExecuteData called with no current XMLParser.") ;
	}

	(*this->Stream).imbue(std::locale::classic());
	this->XMLParser->SetStream(this->Stream);

	if(!this->InformationError)
	{
		this->XMLParser->SetAbort(0);
		this->DataError = 0;

		//this->ReadXMLData(); //  <--- Implement this !
		reader (argument) ;
	}

	this->CloseVTKFile();
}



template <class VtkXmlReaderClass>
template 
<
	class LatticeArrangement, 
	class DataType, 
	TileDataArrangement DataArrangement
>
inline	
void ReaderVtkBase <VtkXmlReaderClass>::
/*
	Look at:		
		vtkXMLDataReader::ReadPieceData()
 */
readTiledLattice
(
	TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
		& tiledLattice
)
{
	typedef TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
		TiledLatticeType ;

	computeNumberOfPoints() ;
	const unsigned nNodes = this->GetNumberOfPoints() ;

	vtkXMLDataElement* ePointData = this->PointDataElements [this->Piece];
	
	{
		auto nodeArray = 
			readDataArray (ePointData, "nodeBaseType", nNodes) ;
		
		compareNodeBaseTypes (tiledLattice, nodeArray) ;
		nodeArray->Delete() ;
	}

	{
		auto nodePlacement = 
			readDataArray (ePointData, "nodePlacementModifier", nNodes) ;

		comparePlacementModifiers (tiledLattice, nodePlacement) ;
		nodePlacement->Delete() ;
	}


#define NODE_MODIFICATOR_FUNCTION                                 \
[&] (typename TiledLatticeType::TileType::DefaultNodeType & node, \
	 	 vtkDoubleArray * dataArray,                    \
	 	 vtkIdType pointId)

	for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
	{
		std::string fArrayName =
			buildFArrayName <LatticeArrangement> ("f", q) ;

		updateTiledLatticeArray (tiledLattice, ePointData, fArrayName, 1,
			
			NODE_MODIFICATOR_FUNCTION
			{
				DataType val = dataArray->GetValue (pointId) ;

				node.f     (q) = val ;
				node.fPost (q) = val ;
			}
		) ;
	}
	tiledLattice.setValidCopyIDToF() ;


	updateTiledLatticeArray (tiledLattice, ePointData, "boundaryVelocityLB",
													 LatticeArrangement::getD(),
		
		NODE_MODIFICATOR_FUNCTION
		{
			for (unsigned i=0 ; i < LatticeArrangement::getD() ; i++)
			{
				node.uBoundary (i) = dataArray->GetComponent (pointId, i) ;
			}
		}
	) ;

	updateTiledLatticeArray (tiledLattice, ePointData, "velocityT0LB", 
													 LatticeArrangement::getD(),
		
		NODE_MODIFICATOR_FUNCTION
		{
			for (unsigned i=0 ; i < LatticeArrangement::getD() ; i++)
			{
				node.uT0 (i) = dataArray->GetComponent (pointId, i) ;
			}
		}
	) ;

	updateTiledLatticeArray (tiledLattice, ePointData, "velocityLB", 
													 LatticeArrangement::getD(),
		
		NODE_MODIFICATOR_FUNCTION
		{
			for (unsigned i=0 ; i < LatticeArrangement::getD() ; i++)
			{
				node.u (i) = dataArray->GetComponent (pointId, i) ;
			}
		}
	) ;

	updateTiledLatticeArray (tiledLattice, ePointData, "rhoLB", 1,
		
		NODE_MODIFICATOR_FUNCTION
		{
			node.rho() = dataArray->GetValue (pointId) ;
		}
	) ;

	updateTiledLatticeArray (tiledLattice, ePointData, "rhoT0LB", 1,

		NODE_MODIFICATOR_FUNCTION
		{
			node.rho0() = dataArray->GetValue (pointId) ;
		}
	) ;

	updateTiledLatticeArray (tiledLattice, ePointData, "boundaryRho", 1,

		NODE_MODIFICATOR_FUNCTION
		{
			node.rhoBoundary() = dataArray->GetValue (pointId) ;
		}
	) ;

#undef NODE_MODIFICATOR_FUNCTION
}



template <class VtkXmlReaderClass>
template <class TiledLattice, class NodeModificator>
inline	
void ReaderVtkBase <VtkXmlReaderClass>::
updateTiledLatticeArray
(
	TiledLattice & tiledLattice,
	vtkXMLDataElement * vtkXmlData,
	const std::string dataName,
	const int requiredNumberOfComponents,
	NodeModificator nodeModificator
)
{
	typedef typename TiledLattice::DataTypeType DataTypeType ;

	const unsigned nNodes = this->GetNumberOfPoints() ;

	auto dataArray = readDataArray (vtkXmlData, dataName, nNodes) ;

	if (dataArray->GetDataType() != VtkTypes <DataTypeType>::VTK_TYPE)
	{
		THROW ("Wrong data type of " + dataName + " in vtk file") ;
	}
	if (requiredNumberOfComponents !=  dataArray->GetNumberOfComponents())
	{
		THROW ("Wrong number of components in " + dataName 
						+ " array in vtk file") ;
	}

	auto castedDataArray = 
		VtkTypes <DataTypeType>::ArrayType::SafeDownCast (dataArray) ;

	resetPointId() ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node,
				 Coordinates & globCoord)
		{
			auto pointId = getAndUpdatePointId (globCoord) ;

			nodeModificator (node, castedDataArray, pointId) ;
		}
	) ;

	dataArray->Delete() ;
}



template <class VtkXmlReaderClass>
template <class TiledLattice>
inline
void ReaderVtkBase <VtkXmlReaderClass>::
compareNodeBaseTypes (TiledLattice & tiledLattice, 
											vtkAbstractArray * dataArray)
{
	auto tStatistic = tiledLattice.getTileLayout().computeTilingStatistic() ;
	unsigned dSize = dataArray->GetNumberOfTuples() ;
	unsigned nNodes = 0 ;

	if (1 == this->IsA ("vtkXMLImageDataReader"))
	{
		nNodes = tStatistic.computeNTotalNodes() ;
	}
	else if (1 == this->IsA ("vtkXMLUnstructuredDataReader"))
	{
		nNodes = tStatistic.getNNodesInNonEmptyTiles() ;
	}
	else
	{
		THROW ("Unimplemented size check") ;
	}
	if (dSize != nNodes)
	{
		std::stringstream ss ;
		ss << "Size of read node base types (" << dSize << ") array differs "
			"from number of nodes in tiled lattice (" << nNodes << ")" ;

		THROW (ss.str().c_str()) ;
	}

	
	if (dataArray->GetDataType() != VTK_UNSIGNED_CHAR)
	{
		THROW ("Wrong type of \"nodeBaseTypes\" array in vtk file") ;
	}

	auto castedArray = vtkUnsignedCharArray::SafeDownCast (dataArray) ;
	
	resetPointId() ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node,
				 Coordinates & globCoord)
		{
			NodeBaseType tLatticeBaseType = node.nodeType().getBaseType() ;

			auto pointId = getAndUpdatePointId (globCoord) ;
			
			NodeBaseType readBaseType = static_cast <NodeBaseType>
				(castedArray->GetValue (pointId)) ;

			if (tLatticeBaseType != readBaseType)
			{
				std::stringstream ss ;
				
				ss << "Lattice node base type differs from node base type in vtk file: "
					 << "current node = " << tLatticeBaseType
					 << " (" << (int)tLatticeBaseType << ")"
					 << ", read node = " << readBaseType
					 << " (" << (int)readBaseType << ")"
					 << " for node at " << globCoord
					 << ", pointId = " << pointId
					 ;
				
				THROW (ss.str()) ;
			}
		}
	) ;
}



template <class VtkXmlReaderClass>
template <class TiledLattice>
inline
void ReaderVtkBase <VtkXmlReaderClass>::
comparePlacementModifiers (TiledLattice & tiledLattice, 
													 vtkAbstractArray * dataArray)
{
	resetPointId() ;
	tiledLattice.forEachNode
	(
		[&] (typename TiledLattice::TileType::DefaultNodeType & node,
				 Coordinates & globCoord)
		{
			auto pointId = getAndUpdatePointId (globCoord) ;

			unsigned char tLatticePlacementModifier = 
				static_cast<unsigned char> (node.nodeType().getPlacementModifier()) ;
			
			unsigned char readPlacementModifier =
				dataArray->GetVariantValue (pointId).ToUnsignedChar() ;

			if (tLatticePlacementModifier != readPlacementModifier)
			{
				THROW ("Lattice node placement modifiers differs from "
							 "node placement modifiers in vtk file") ;
			}
		}
	) ;
}



inline
int ReaderVtkImage::
ReadPrimaryElement (vtkXMLDataElement* ePrimary) 
{
	// This one crashes !
	// The problem is, when vtkXMLStructuredDataReader calls 
	//	GetCurrentOutputInformation()
	//return vtkXMLStructuredDataReader::ReadPrimaryElement (ePrimary) ;
	//return vtkXMLImageDataReader::ReadPrimaryElement (ePrimary) ;


	/*
		 Look at:
		 vtkXMLImageDataReader::ReadPrimaryElement ()
	 */
	if(ePrimary->GetVectorAttribute("Origin", 3, this->Origin) != 3)
	{
		this->Origin[0] = 0;
		this->Origin[1] = 0;
		this->Origin[2] = 0;
	}
	if(ePrimary->GetVectorAttribute("Spacing", 3, this->Spacing) != 3)
	{
		this->Spacing[0] = 1;
		this->Spacing[1] = 1;
		this->Spacing[2] = 1;
	}
	/*
		 Look at:
		 vtkXMLStructuredDataReader::ReadPrimaryElement()
	 */
	// Get the whole extent attribute.
	int extent[6];
	if(ePrimary->GetVectorAttribute("WholeExtent", 6, extent) == 6)
	{
		memcpy(this->WholeExtent, extent, 6*sizeof(int));

		// Check each axis to see if it has cells.
		for(int a=0; a < 3; ++a)
		{
			this->AxesEmpty[a] = (extent[2*a+1] > extent[2*a])? 0 : 1;
		}
	}
	else
	{
		vtkErrorMacro(<< this->GetDataSetName() << " element has no WholeExtent.");
		return 0;
	}

	if(ePrimary->GetVectorAttribute("PhysicalOrigin", 3, this->physicalOrigin_) != 3)
	{
		this->physicalOrigin_ [0] = NAN ;
		this->physicalOrigin_ [1] = NAN ;
		this->physicalOrigin_ [2] = NAN ;
	}
	if(ePrimary->GetScalarAttribute("PhysicalSpacing", this->physicalSpacing_) != 1)
	{
		this->physicalSpacing_ = NAN ;
	}

	vtkXMLDataReader::ReadPrimaryElement (ePrimary) ;


	return 1;
}



inline
vtkXMLDataElement * ReaderVtkImage::
getXmlPointData ()
{
	return this->PointDataElements [this->Piece] ;
}



inline
void ReaderVtkImage::
resetPointId()
{}



inline
vtkIdType ReaderVtkImage::
getPointId (Coordinates const & pointCoordinates) 
{ 
	int ijk[3] ;
	ijk[0] = pointCoordinates.getX() ;
	ijk[1] = pointCoordinates.getY() ;
	ijk[2] = pointCoordinates.getZ() ;

	return vtkStructuredData::ComputePointIdForExtent
							(this->WholeExtent, ijk) ;
}



inline
vtkIdType ReaderVtkImage::
getAndUpdatePointId (Coordinates const & pointCoordinates) 
{ 
	return getPointId (pointCoordinates) ;
}



inline
void ReaderVtkImage::
computeNumberOfPoints()
{
	this->ComputePointDimensions (this->WholeExtent, this->PointDimensions) ;
}



inline
vtkXMLDataElement * ReaderVtkUnstructured::
getXmlPointData ()
{
	return this->PointDataElements [this->Piece] ;
}



inline
void ReaderVtkUnstructured::
resetPointId()
{ 
	pointId_ = 0 ; 
}



inline
vtkIdType ReaderVtkUnstructured::
getPointId (Coordinates const & pointCoordinates __attribute__((unused)))
{
	return pointId_ ; 
}



inline
vtkIdType ReaderVtkUnstructured::
getAndUpdatePointId (Coordinates const & pointCoordinates __attribute__((unused))) 
{
	return pointId_++ ; 
}



inline
void ReaderVtkUnstructured::
computeNumberOfPoints()
{
	//	Look at:
	//		vtkXMLUnstructuredDataReader::SetupOutputTotals()
  this->TotalNumberOfPoints = 0;
  for(int i=0; i < this->NumberOfPieces; ++i)
    {
    this->TotalNumberOfPoints += this->NumberOfPoints[i];
    }  
  this->StartPoint = 0;
}



}



#endif
