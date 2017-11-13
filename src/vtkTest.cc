#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <sstream>
#include <iostream>

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkIndent.h>
#include <vtkArrayRange.h>
#include <vtkDataArray.h>
#include <vtkDataArrayTemplate.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkImageData.h>
#include <vtkXMLImageDataReader.h>
#include <vtkIndent.h>
#include <vtkFieldData.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkErrorCode.h>
#include <vtkImageData.h>
#include <vtkXMLImageDataWriter.h>



using namespace std ;



class MyImageDataWriter : public vtkXMLImageDataWriter
{
	public:

		int myWrite (vtkSmartPointer <vtkDoubleArray> dataArray) ;
		void registerOstream (ostream & os) 
		{
			Stream = &os ;
		}
} ;


/*
	Based on vtkXMLStructuredDataWriter::ProcessRequest(...)
*/
int MyImageDataWriter::
myWrite (vtkSmartPointer <vtkDoubleArray> dataArray)
{

	SetDataModeToBinary() ;     // !!!!
	SetCompressorTypeToNone() ; // !!!!


	OpenFile() ;
	StartFile() ;


	/*
		Look at:
			vtkXMLStructuredDataWriter::WriteHeader()
			vtkXMLWriter::WritePrimaryElement(ostream &os, vtkIndent indent)
			vtkXMLWriter::WritePrimaryElement(ostream &os, vtkIndent indent)
			vtkXMLImageDataWriter::WritePrimaryElementAttributes(ostream &os, vtkIndent indent)
	*/
  vtkIndent indent = vtkIndent().GetNextIndent();
  ostream& os = *(this->Stream);
	int geometryExtent[6] = {0,1, 0,1, 0,1} ;
	double Origin[3] = {0,0,0} ;
	double Spacing[3] = {1,1,1} ;

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
	vtkIndent indent2 = indent.GetNextIndent();

	os << indent << "<Piece";
		this->WriteVectorAttribute("Extent", 6, geometryExtent);
	os << ">\n";
	

  os << indent2 << "<PointData";
  os << ">\n";
		this->WriteArrayInline(dataArray, indent2.GetNextIndent(), dataArray->GetName()) ;
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

	return 1 ;
}



TEST (vtk,1)
{
	const unsigned xSize = 2 ;
	const unsigned ySize = 2 ;
	const unsigned zSize = 2 ;

	auto savedData = vtkSmartPointer <vtkStructuredPoints>::New() ;
	savedData->SetOrigin (0,0,0) ;
	savedData->SetSpacing (1,1,1) ; // dxPhys, dxPhys, dxPhys
	savedData->SetDimensions (xSize, ySize, zSize) ; 

	const int nPoints = savedData->GetNumberOfPoints() ;

	auto data_1 = vtkSmartPointer <vtkDoubleArray>::New() ;
	data_1->SetName ("data_1") ;
	data_1->SetNumberOfComponents (1) ;
	data_1->SetNumberOfTuples (nPoints) ;
  
	cout << "nTuples=" << data_1->GetNumberOfTuples() << "\n" ;

	double vTmp = 0 ;
	for (vtkIdType tupleIdx=0 ; tupleIdx < data_1->GetNumberOfTuples() ; tupleIdx++)
		{
			data_1->SetComponent (tupleIdx, 0, vTmp) ;
			vTmp += 1.0 ;
		}
	
	savedData->GetPointData()->AddArray (data_1) ;



  auto writer = new MyImageDataWriter ;

	stringstream origStream ;
	writer->registerOstream (origStream) ;

	writer->SetCompressorTypeToNone() ;
	writer->SetDataModeToBinary() ;
  writer->SetInput (savedData);
  writer->Write();	

  auto writer2 = new MyImageDataWriter ;

	stringstream myStream ;
	writer2->registerOstream (myStream) ;

	writer2->myWrite (data_1) ;

	writer->Delete() ;
	writer2->Delete() ;

	cout << "\nOriginal stream:\n\n" << origStream.str() << "\n" ;
	cout << "\nMy stream:\n\n" << myStream.str() << "\n" ;

	EXPECT_EQ (origStream.str(), myStream.str()) ;
}
