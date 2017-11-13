#include <vtkObjectFactory.h>

#include "ReaderVtk.hpp"



using namespace microflow ;



vtkStandardNewMacro (ReaderVtkImage) ;
vtkStandardNewMacro (ReaderVtkUnstructured) ;



void ReaderVtkImage::
readNodeLayout (NodeLayout & nodeLayout)
{
	readSkeleton (nodeLayout,

		[&] (NodeLayout & nodeLayout)
		{
			computeNumberOfPoints() ;
			const unsigned nNodes = this->GetNumberOfPoints() ;

			Size sizeInNodes 
				(PointDimensions[0], PointDimensions[1], PointDimensions[2]) ;

			logger << "Reading geometry from \"" << GetFileName() << "\""
						 << ", found " << nNodes << " nodes " << sizeInNodes
						 << "\n" ;

			nodeLayout.resizeWithContent (sizeInNodes) ;

			vtkXMLDataElement* ePointData = this->PointDataElements [this->Piece];
			
			auto nodeArray = 
				readDataArray (ePointData, "nodeType", nNodes) ;

			for (unsigned z=0 ; z < sizeInNodes.getDepth() ; z++)
				for (unsigned y=0 ; y < sizeInNodes.getHeight() ; y++)
					for (unsigned x=0 ; x < sizeInNodes.getWidth() ; x++)
					{
						vtkIdType nodeIdx = XYZ::linearize (x,y,z, 
																				sizeInNodes.getWidth(), 
																				sizeInNodes.getHeight(), 
																				sizeInNodes.getDepth()) ;

						unsigned char nodeValue =
							vtkUnsignedCharArray::SafeDownCast (nodeArray)->GetValue (nodeIdx) ;

						if (1 == nodeValue)
						{
							nodeLayout.setNodeType (x,y,z, NodeBaseType::FLUID) ;
						}
						else
						{
							nodeLayout.setNodeType (x,y,z, NodeBaseType::SOLID) ;
						}
					}
			
			nodeArray->Delete() ;
		}
	) ;
}
