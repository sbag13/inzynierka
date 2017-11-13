#ifndef READER_VTK_HPP
#define READER_VTK_HPP



#include <sstream>

#include <vtkAbstractArray.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLStructuredGridReader.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLDataParser.h>
#include <vtkInformation.h>
#include <vtkSetGet.h>

#include "Direction.hpp"
#include "TiledLattice.hpp"
#include "BaseIO.hpp"
#include "VtkTypes.hpp"



namespace microflow
{



template <class VtkXmlReaderClass>
class ReaderVtkBase : public VtkXmlReaderClass
{
	public:
	
		void registerIstream (std::istream & is) ;

		template 
		<
			class LatticeArrangement, 
			class DataType, 
			TileDataArrangement DataArrangement
		>
		void read 
		(
			TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
				& tiledLattice
		) ;
		

	protected:

		virtual
		vtkXMLDataElement * getXmlPointData() = 0 ;


		/*
			TODO: Move to separate class, maybe template <VtkXmlReaderClass>.
						Next use this class in WriterVtk too.
		*/
		virtual
		void resetPointId() = 0 ;
		virtual
		vtkIdType getPointId (Coordinates const & pointCoordinates) = 0 ;
		virtual
		vtkIdType getAndUpdatePointId (Coordinates const & pointCoordinates) = 0 ;


		virtual
		void computeNumberOfPoints() = 0 ;


		template <class Argument, class Reader>
		void readSkeleton (Argument & argument, Reader reader) ;

		template 
		<
			class LatticeArrangement, 
			class DataType, 
			TileDataArrangement DataArrangement
		>
		void readTiledLattice
		(
			TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
				& tiledLattice
		) ;
		
		template <class TiledLattice, class NodeModificator>
		void updateTiledLatticeArray
		(
			TiledLattice & tiledLattice,
			vtkXMLDataElement * vtkXmlData,
			const std::string dataName,
			const int requiredNumberOfComponents,
			NodeModificator nodeModificator
		) ;
		
		vtkXMLDataElement * 
		findArrayDataElement (vtkXMLDataElement * dataElements, std::string name) ;

		//TODO: Use vtkSmartPointer <vtkAbstractArray>.
		vtkAbstractArray *
		readDataArray (vtkXMLDataElement * dataElements,
									 std::string name, unsigned nValues) ;

		template <class TiledLattice>
		void compareNodeBaseTypes (TiledLattice & tiledLattice, 
															 vtkAbstractArray * dataArray) ;

		template <class TiledLattice>
		void comparePlacementModifiers (TiledLattice & tiledLattice, 
															 			vtkAbstractArray * dataArray) ;
} ;



class ReaderVtkImage 
	: public ReaderVtkBase <vtkXMLImageDataReader>
{
	public:

		static ReaderVtkImage * New() ;
		vtkTypeMacro (ReaderVtkImage, vtkXMLImageDataReader) ;
		
		void readNodeLayout (NodeLayout & nodeLayout) ;

		UniversalCoordinates<double> getPhysicalOrigin() const 
		{
			return UniversalCoordinates<double> 
				(physicalOrigin_[0], physicalOrigin_[1], physicalOrigin_[2])  ;
		}

		double getPhysicalSpacing() const
		{
			return physicalSpacing_ ;
		}


	protected:

		virtual
		vtkXMLDataElement * getXmlPointData() ;

		int ReadPrimaryElement (vtkXMLDataElement* ePrimary) ;

		virtual
		void resetPointId() ;
		virtual
		vtkIdType getPointId (Coordinates const & pointCoordinates) ;
		virtual
		vtkIdType getAndUpdatePointId (Coordinates const & pointCoordinates) ;

		virtual
		void computeNumberOfPoints() ;

	
	private:
		
		double physicalOrigin_ [3] ;
		double physicalSpacing_ ;
} ;



class ReaderVtkUnstructured
	: public ReaderVtkBase <vtkXMLUnstructuredGridReader>
{
	public:

		static ReaderVtkUnstructured * New() ;
		vtkTypeMacro (ReaderVtkUnstructured, vtkXMLUnstructuredGridReader) ;


	protected:

		virtual
		vtkXMLDataElement * getXmlPointData() ;

		virtual
		void resetPointId() ;
		virtual
		vtkIdType getPointId (Coordinates const & pointCoordinates) ;
		virtual
		vtkIdType getAndUpdatePointId (Coordinates const & pointCoordinates) ;


		virtual
		void computeNumberOfPoints() ;

	
	private:
	
		vtkIdType pointId_ ;
} ;



}



#include "ReaderVtk.hh"



#endif
