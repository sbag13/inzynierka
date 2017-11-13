#ifndef WRITER_HPP
#define WRITER_HPP


#include <string>

#include "TiledLattice.hpp"



namespace microflow
{



template 
<
	class LatticeArrangement, 
	class DataType, 
	TileDataArrangement DataArrangement
>
class Writer
{
	public:

		typedef TiledLattice <LatticeArrangement,DataType,
													StorageOnCPU,DataArrangement> TiledLatticeType ;

		Writer (TiledLatticeType & tiledLattice) ;

		template <class Settings>
		void saveVtk 
		(
			const Settings & settings,
			const std::string & filePath
		) const ;


		template <class Settings>
		size_t estimateDataSizeForUnstructuredGrid (const Settings & settings) const ;

		template <class Settings>
		size_t estimateDataSizeForStructuredGrid (const Settings & settings) const ;

		template <class Settings>
		unsigned estimateBytesPerNode (const Settings & settings) const ;

	
	private:

		template <class Settings, class VtkWriter>
		void saveVtkHelper
		(
			const Settings & settings,
			const std::string & filePath,
			VtkWriter & vtkWriter
		) const ;
		
		// TODO: Make it const.
		TiledLatticeType & tiledLattice_ ;
} ;



}



#include "Writer.hh"



#endif
