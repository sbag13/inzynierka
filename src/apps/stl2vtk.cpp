/***************************************************************************
 *   Copyright (C) 2011 by F. P. Beekhof                                   *
 *   fpbeekhof@gmail.com                                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with program; if not, write to the                              *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <string>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>

#include <vtkIndent.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkLongLongArray.h>
#include <vtkUnsignedLongLongArray.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkStructuredData.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkCellType.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

#include "gzstream.h"
#include "MultidimensionalMappers.hpp"
#include "NodeBaseType.hpp"
#include "PerformanceMeter.hpp"
#include "microflowTools.hpp"



using namespace cvmlcpp;
using namespace microflow ;
using namespace std ;



typedef float T;

/*
	TODO: Use of Matrix class results in limit to 10^9 nodes. This limit
				may be removed by using either DTree or CompressedMatrix, but
				for now it is unclear, ho to do it.
*/
typedef Matrix <char, 3> VoxelMatrix ;



class WriterVtk : public vtkXMLImageDataWriter
{
	public:
		static WriterVtk * New() ;
		vtkTypeMacro (WriterVtk, vtkXMLImageDataWriter) ;

		void write 
		(
			Geometry<T> & geometry, 
			VoxelMatrix & voxels,
			T voxelSize, 
			std::string filePath
		)
		{
			PerformanceMeter pm (1) ;

			int geometryExtent[6] = {0,-1, 0,-1, 0,-1} ;
			geometryExtent [0] = 0 ;
			geometryExtent [2] = 0 ;
			geometryExtent [4] = 0 ;
			geometryExtent [1] = voxels.extent(0) ;
			geometryExtent [3] = voxels.extent(1) ;
			geometryExtent [5] = voxels.extent(2) ;

			T origin[3] = {0,0,0} ;

			T physicalOrigin [3] ;
			physicalOrigin [0] = geometry.min(0) ;
			physicalOrigin [1] = geometry.min(1) ;
			physicalOrigin [2] = geometry.min(2) ;

			T spacing[3] = {1, 1, 1} ;


			auto const widthInNodes  = voxels.extent (0) + 1 ;
			auto const heightInNodes = voxels.extent (1) + 1 ;
			auto const depthInNodes  = voxels.extent (2) + 1 ;

			const size_t nNodes = 
				static_cast<size_t>(widthInNodes) *
				static_cast<size_t>(heightInNodes) *
				static_cast<size_t>(depthInNodes) ;

			
			pm.start() ;
			cout << "Marking nodes..." << flush ;

			auto nodeBaseType = vtkSmartPointer <vtkUnsignedCharArray>::New() ;
			nodeBaseType->SetName ("nodeType") ;
			nodeBaseType->SetNumberOfComponents (1) ;
			nodeBaseType->SetNumberOfTuples (nNodes) ;
			nodeBaseType->FillComponent 
				(0, static_cast<unsigned char>(NodeBaseType::SOLID)) ;


			vtkIdType planeSize = widthInNodes * heightInNodes ;

			for (int z=0 ; z < voxels.extent (2) ; z++)
				for (int y=0 ; y < voxels.extent (1) ; y++)
					for (int x=0 ; x < voxels.extent (0) ; x++)
					{

						vtkIdType nodeIndex = XYZ::linearize (x,y,z,
																	 widthInNodes, heightInNodes, depthInNodes) ;

						if (1 == voxels[x][y][z])
						{
							nodeBaseType->SetTuple1 (nodeIndex, 
								static_cast<double>(NodeBaseType::FLUID)) ;
							nodeBaseType->SetTuple1 (nodeIndex + 1, 
								static_cast<double>(NodeBaseType::FLUID)) ;
							nodeBaseType->SetTuple1 (nodeIndex + widthInNodes, 
								static_cast<double>(NodeBaseType::FLUID)) ;
							nodeBaseType->SetTuple1 (nodeIndex + widthInNodes + 1, 
								static_cast<double>(NodeBaseType::FLUID)) ;

							nodeBaseType->SetTuple1 (planeSize + nodeIndex, 
								static_cast<double>(NodeBaseType::FLUID)) ;
							nodeBaseType->SetTuple1 (planeSize + nodeIndex + 1, 
								static_cast<double>(NodeBaseType::FLUID)) ;
							nodeBaseType->SetTuple1 (planeSize + nodeIndex + widthInNodes, 
								static_cast<double>(NodeBaseType::FLUID)) ;
							nodeBaseType->SetTuple1 (planeSize + nodeIndex + widthInNodes + 1, 
								static_cast<double>(NodeBaseType::FLUID)) ;
						}
					}

			pm.stop() ;
			cout << "OK, took " << microsecondsToHuman (pm.findMaxDuration()) << ".\n" ;
			pm.clear() ;


			filePath += "." ;
			filePath += GetDefaultFileExtension() ;

			cout << "Writing \"" << filePath << "\"..." << flush ;
			pm.start() ;

			SetFileName (filePath.c_str()) ;


			OpenFile() ;
			StartFile() ;

			vtkIndent indent = vtkIndent().GetNextIndent();
			ostream& os = *(this->Stream);


			os << indent << "<" << this->GetDataSetName();
				this->WriteVectorAttribute("WholeExtent", 6, geometryExtent);
				this->WriteVectorAttribute("Origin", 3, origin);
				this->WriteVectorAttribute("Spacing", 3, spacing);

		  	this->WriteVectorAttribute ("PhysicalOrigin", 3, physicalOrigin);
				this->WriteScalarAttribute ("PhysicalSpacing", voxelSize) ;
		  os << ">\n";

			os << indent << "<Piece";
				this->WriteVectorAttribute("Extent", 6, geometryExtent);
			os << ">\n";
			

			vtkIndent indent2 = indent.GetNextIndent();
			os << indent2 << "<PointData";
			os << ">\n";

				this->WriteArrayInline (nodeBaseType, indent2.GetNextIndent()) ;

			os << indent2 << "</PointData>\n";


			os << indent2 << "<CellData";
			os << ">\n";
			os << indent2 << "</CellData>\n";

			os << indent << "</Piece>\n";


			os << indent << "</" << this->GetDataSetName() << ">\n";

			EndFile() ;
			CloseFile() ;

			pm.stop() ;
			cout << "OK, took " << microsecondsToHuman (pm.findMaxDuration()) << ".\n" ;
			pm.clear() ;
		}
} ;



vtkStandardNewMacro (WriterVtk) ;



void usage(const char * const name)
{
	std::cout << "usage: "<<name<< " <stl-file> <voxelsize>" << std::endl;
	exit(0);
}

int main(int argc, const char * const argv[])
{
	PerformanceMeter pmGlobal (1) ;
	PerformanceMeter pm (1) ;

	pmGlobal.start() ;

	if (argc != 3)
		usage(argv[0]);

	const T voxel_size = std::atof(argv[2]);
	if (!(voxel_size > 0))
	{
		std::cout << "ERROR: voxel size found as "<<voxel_size<<
		", should be positive float value. Bailing out.\n"<<std::endl;
		usage(argv[0]);
		return -1;
	}

	cout << "Reading \"" << argv[1] << "\"..." << flush ;
	pm.start() ;
		Geometry<T> geometry;
		if (!readSTL(geometry, argv[1]))
		{
			std::cout << "ERROR: can't read STL file ["<<argv[1]<<"], bailing out.\n"<<std::endl;
			return -1;
		}
	pm.stop() ;
	cout << "OK, took " << microsecondsToHuman (pm.findMaxDuration()) << ".\n" ;
	pm.clear() ;

	cout << "Voxelisation..." << flush ;
	pm.start() ;
		
		VoxelMatrix voxels ;

		if (!voxelize(geometry, voxels, voxel_size))
		{
			std::cout << "ERROR: voxelization failed, bailing out.\n"<<std::endl;
			return -1;
		}
	pm.stop() ;
	cout << "OK, generated mesh " 
		 	 << voxels.extent(0) << "x" << voxels.extent(1) << "x" << voxels.extent(2)
		   << ", took " << microsecondsToHuman (pm.findMaxDuration())
		 	 << ".\n" ;
	pm.clear() ;
	


	const std::string input(argv[1]);
	std::string output;
	if (argc == 5)
		output = argv[4];
	else if (input.length() > 5)
		output = input.substr(0, input.length() - 4); // strip extention
	else
		output = input;



	auto writer = vtkSmartPointer <WriterVtk>::New() ;

	writer->SetCompressorTypeToZLib() ;
	writer->SetDataModeToBinary() ;

	writer->write (geometry, voxels, voxel_size, output) ;


	pmGlobal.stop() ;
	std::cout << "All complete, took " 
						<< microsecondsToHuman (pmGlobal.findMaxDuration()) << ".\n" ;

	return 0;
}
