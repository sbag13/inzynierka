#ifndef WRITER_VTK_HPP
#define WRITER_VTK_HPP


#include <vtkXMLImageDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

#include "Direction.hpp"
#include "TiledLattice.hpp"
#include "BaseIO.hpp"


namespace microflow
{


/*
	Helper class, which modifies the behavior of vtXMLWriter classes.
*/
template <class VtkXmlWriterClass>
class WriterVtkBase : public VtkXmlWriterClass
{
	public:

		void registerOstream (std::ostream & os) ;

		vtkSmartPointer <vtkDoubleArray> allocateArray1D (unsigned nElements) ;
		vtkSmartPointer <vtkDoubleArray> allocateArray3D (unsigned nElements) ;


	protected:

		typedef VtkXmlWriterClass BaseClass ;
	
		// WARNING: Our implementation of partial VTK writing works only for 
		//          inline writing (ascii/binary mode). Thus we need to disable
		//					switching to appended mode.
	  void SetDataModeToAppended() ;
		inline void SetDataMode (int mode) ;

		// Helper methods useful while saving VTK files with TiledLattice.
		void fillArray (vtkAbstractArray * dataArray, double value) ;

		template <class TiledLattice, class Functor>
		void forEachNode (TiledLattice const & tiledLattice, Functor functor) ;
} ;



class WriterVtkImage 
	: public WriterVtkBase <vtkXMLImageDataWriter>
{
	public:
	  
		static WriterVtkImage * New() ;
		vtkTypeMacro (WriterVtkImage, vtkXMLImageDataWriter) ;

		template 
		<
			class LatticeArrangement, 
			class DataType, 
			TileDataArrangement DataArrangement,
			class Settings
		>
		// TODO: Make this method not a template, LatticeArrangement parameters 
		//			 may be obtained from some base abstract class.
		void write 
		(
			TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
				& tiledLattice,
			Settings const & settings
		) ;
} ;



class WriterVtkUnstructured
	: public WriterVtkBase <vtkXMLUnstructuredGridWriter>
{
	public:
	  
		static WriterVtkUnstructured * New() ;
		vtkTypeMacro (WriterVtkUnstructured, vtkXMLUnstructuredGridWriter) ;

		template 
		<
			class LatticeArrangement, 
			class DataType, 
			TileDataArrangement DataArrangement,
			class Settings
		>
		// TODO: Make this method not a template, LatticeArrangement parameters 
		//			 may be obtained from some base abstract class.
		void write 
		(
			TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
				& tiledLattice,
			Settings const & settings
		) ;


		// TODO: Similar method may be useful for WriterVtkImage. The unfinished
		//			 code is in WriterTest.cc file.
		template 
		<
			class LatticeArrangement, 
			class DataType, 
			TileDataArrangement DataArrangement,
			class Settings
		>
		vtkSmartPointer <vtkUnstructuredGrid> buildUnstructuredGrid
		(
			TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
				& tiledLattice,
			Settings const & settings
		) ;

		template 
		<
			class LatticeArrangement, 
			class DataType, 
			TileDataArrangement DataArrangement,
			class Settings
		>
		void addDataToGrid
		(
			vtkSmartPointer <vtkUnstructuredGrid> & grid,
			TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement>
				& tiledLattice,
			const Settings & settings
		) ;


	private:

		template <class Tile>
		void addCellsFromTile
		(
			Tile & tile, vtkSmartPointer <vtkUnstructuredGrid> & grid
		) ;

		template <class Tile>
		void addCellsFromPlaneB
		(
		 	Tile & tile,
			vtkSmartPointer <vtkUnstructuredGrid> & grid
		) ;

		template <class Tile>
		void addCellsFromPlaneW
		(
			Tile & tile,
			vtkSmartPointer <vtkUnstructuredGrid> & grid
		) ;

		template <class Tile>
		void addCellsFromPlaneS
		(
			Tile & tile,
			vtkSmartPointer <vtkUnstructuredGrid> & grid
		) ;

		template <class Tile>
		void addCellsFromEdgeWB
		(
			Tile & tile,
			vtkSmartPointer <vtkUnstructuredGrid> & grid
		) ;

		template <class Tile>
		void addCellsFromEdgeSB
		(
			Tile & tile,
			vtkSmartPointer <vtkUnstructuredGrid> & grid
		) ;

		template <class Tile>
		void addCellsFromEdgeSW
		(
			Tile & tile,
			vtkSmartPointer <vtkUnstructuredGrid> & grid
		) ;

		template <class Tile>
		void addCellFromCorner
		(
			Tile & tile,
			vtkSmartPointer <vtkUnstructuredGrid> & grid
		) ;

		template <class Tile>
		unsigned computeGlobalIndex (Tile & tile, 
																 unsigned nodeInTileX, 
																 unsigned nodeInTileY, 
																 unsigned nodeInTileZ) ;

		template <class TiledLattice>
		void buildNodeArrays 
		(
			vtkSmartPointer <vtkUnsignedCharArray> & nodeBaseType,
			vtkSmartPointer <vtkUnsignedCharArray> & nodePlacementModifier,
			TiledLattice & tiledLattice
		) ;

		template <class TiledLattice>
		void buildBoundaryVelocityLBArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice
		) ;

		template <class TiledLattice>
		void buildVelocityLBArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice
		) ;

		template <class TiledLattice>
		void buildVelocityT0LBArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice
		) ;

		template <class TiledLattice, class Settings>
		void buildBoundaryVelocityPhysicalArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice,
			const Settings & settings
		) ;

		template <class TiledLattice, class Settings>
		void buildVelocityPhysicalArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice,
			const Settings & settings
		) ;

		template <class TiledLattice>
		void buildBoundaryRhoLBArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice
		) ;

		template <class TiledLattice>
		void buildRhoLBArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice
		) ;

		template <class TiledLattice>
		void buildRhoT0LBArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice
		) ;

		template <class TiledLattice, class Settings>
		void buildPressureArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice,
			const Settings & settings
		) ;

		template <class TiledLattice>
		void buildFArray
		(
			vtkSmartPointer <vtkDoubleArray> & dataArray,
			TiledLattice & tiledLattice,
			Direction::DirectionIndex q
		) ;

		
} ;



}



#include "WriterVtk.hh"



#endif
