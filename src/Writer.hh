#ifndef WRITER_HH
#define WRITER_HH



#include "WriterVtk.hpp"
#include "TilingStatistic.hpp"



namespace microflow
{



#define TEMPLATE_WRITER                \
template                               \
<                                      \
	class LatticeArrangement,            \
	class DataType,                      \
	TileDataArrangement DataArrangement  \
>



#define WRITER    \
Writer<LatticeArrangement, DataType, DataArrangement>



TEMPLATE_WRITER
inline
WRITER::
Writer (TiledLatticeType & tiledLattice) 
: tiledLattice_ (tiledLattice) 
{
}



TEMPLATE_WRITER
template <class Settings>
inline
size_t WRITER::
estimateDataSizeForStructuredGrid (const Settings & settings) const
{
	size_t bytesPerNode = estimateBytesPerNode (settings) ;

	size_t nNodes = tiledLattice_.getTileLayout().getNodeLayout().getSize().computeVolume() ;

	return bytesPerNode * nNodes ;
}



TEMPLATE_WRITER
template <class Settings>
inline
size_t WRITER::
estimateDataSizeForUnstructuredGrid (const Settings & settings) const
{
	auto const tilingStatistic = tiledLattice_.getTileLayout().computeTilingStatistic() ;

	size_t nNodes = tilingStatistic.getNNodesInNonEmptyTiles() ;
	// We need additionally to store 3 coordinates per each node.
	size_t bytesPerNode = estimateBytesPerNode (settings) + 3 * sizeof (float) ;

	// Approximate value, the approximation error may be large.
	size_t nCells = tilingStatistic.getNNonSolidNodes() ;
	// Each cell requires: 
	//	1 byte for cell type
	//	1 int64 for cell offset
	//	6 int64 for cell corners positions
	size_t bytesPerCell = 57 ;

	return nNodes * bytesPerNode  +  nCells * bytesPerCell ;
}



TEMPLATE_WRITER
template <class Settings>
inline
unsigned WRITER::
estimateBytesPerNode (const Settings & settings) const
{
	unsigned bytesPerNode = 0 ;
	unsigned bytesPerData = sizeof (DataType) ;
	unsigned q = LatticeArrangement::getQ() ;
	unsigned d = LatticeArrangement::getD() ;

	if (settings.shouldSaveNodes                  ()) bytesPerNode += 2 ;
	if (settings.shouldSaveVelocityPhysical       ()) bytesPerNode += 2 * d * bytesPerData ;
	if (settings.shouldSaveVelocityLB             ()) bytesPerNode += 3 * d * bytesPerData ;
	if (settings.shouldSaveVolumetricMassDensityLB()) bytesPerNode += 2 * bytesPerData ;
	if (settings.shouldSavePressurePhysical       ()) bytesPerNode += bytesPerData ;
	if (settings.shouldSaveMassFlowFractions      ()) bytesPerNode += q * bytesPerData ;

	return bytesPerNode ;
}



TEMPLATE_WRITER
template <class Settings>
inline
void WRITER::
saveVtk (const Settings & settings, const std::string & filePath) const
{
	// WARNING: Extremely oversimplified, does not correctly compute number of cells
	//					and ignores compression impact and XML file overhead (headers etc.).
	size_t unstructuredSize = estimateDataSizeForUnstructuredGrid (settings) ;
	size_t structuredSize   = estimateDataSizeForStructuredGrid   (settings) ;

	logger << "Estimated uncompressed file size " ;

	if (structuredSize > unstructuredSize)
	{
		logger << "(unstructured) : " << bytesToHuman (unstructuredSize) << "\n" ;

		auto writer = vtkSmartPointer <WriterVtkUnstructured>::New() ;
		saveVtkHelper (settings, filePath, *writer) ;
	}
	else
	{
		logger << "(structured) : " << bytesToHuman (structuredSize) << "\n" ;

		auto writer = vtkSmartPointer <WriterVtkImage>::New() ;
		saveVtkHelper (settings, filePath, *writer) ;
	}

	logger << "\n" ;
}



TEMPLATE_WRITER
template <class Settings, class VtkWriter>
inline
void WRITER::
saveVtkHelper 
(
	const Settings & settings, 
	const std::string & filePath,
	VtkWriter & vtkWriter
) const
{
	std::string fullFilePath = filePath + "." + vtkWriter.GetDefaultFileExtension() ;
	
	logger << "Saving " << fullFilePath << "\n" ;

	vtkWriter.SetFileName (fullFilePath.c_str()) ;

	vtkWriter.SetCompressorTypeToZLib() ;
	vtkWriter.SetDataModeToBinary() ;

	vtkWriter.write (tiledLattice_, settings) ;
}



#undef WRITER
#undef TEMPLATE_WRITER


	
}



#endif
