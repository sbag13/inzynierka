#ifndef SIMULATION_ENGINE_HH
#define SIMULATION_ENGINE_HH



#include <sstream>

#include <vtkErrorCode.h>

#include "Writer.hpp"
#include "ReaderVtk.hpp"
#include "fileUtils.hpp"
#include "CheckpointSettings.hpp"



namespace microflow
{



#define TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE         \
template<                                                      \
					class LatticeArrangement,                            \
					template <class, class> class FluidModel,            \
					class CollisionModel,                                \
					class DataType,                                      \
					class ComputationalEngine                            \
				>



#define SIMULATION_ENGINE_SPECIALIZATION_BASE                  \
SimulationEngineSpecializationBase                             \
<                                                              \
	LatticeArrangement, FluidModel, CollisionModel, DataType,    \
	ComputationalEngine                                          \
>



#define TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION              \
template<                                                      \
					class LatticeArrangement,                            \
					template <class, class> class FluidModel,            \
					class CollisionModel,                                \
					class DataType                                       \
				>



#define SIMULATION_ENGINE_SPECIALIZATION                       \
SimulationEngineSpecialization                                 \
<                                                              \
	LatticeArrangement, FluidModel, CollisionModel, DataType,    \
	ComputationalEngineCPU                                       \
>



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
SIMULATION_ENGINE_SPECIALIZATION_BASE::
SimulationEngineSpecializationBase( TileLayout<StorageOnCPU> & tileLayout,
																		ExpandedNodeLayout & expandedNodeLayout,
																		const Settings & settings )
: tiledLatticeCPU_( tileLayout, expandedNodeLayout, settings )
{
	DataType rho0LB = settings.getInitialVolumetricMassDensityLB() ;
	DataType u0LB[LatticeArrangement::getD()] ;
	DataType tau = settings.getTau() ;

	for (unsigned i=0 ; i < LatticeArrangement::getD() ; i++)
	{
		u0LB[i] = settings.getInitialVelocityLB( toAxis(i) ) ;
	}

	latticeCalculator_.reset
	(
		new LatticeCalculatorType (rho0LB, u0LB, tau,
															 settings.getDefaultExternalEdgePressureNode(),
															 settings.getDefaultExternalCornerPressureNode()
															)
	) ;
	latticeCalculatorCPU_.reset
	(
		new LatticeCalculatorTypeCPU (rho0LB, u0LB, tau,
																  settings.getDefaultExternalEdgePressureNode(),
																  settings.getDefaultExternalCornerPressureNode()
																 )
	) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
typename SIMULATION_ENGINE_SPECIALIZATION_BASE::LatticeCalculatorType &
SIMULATION_ENGINE_SPECIALIZATION_BASE::
getLatticeCalculator()
{
	return *latticeCalculator_ ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
typename SIMULATION_ENGINE_SPECIALIZATION_BASE::LatticeCalculatorTypeCPU &
SIMULATION_ENGINE_SPECIALIZATION_BASE::
getLatticeCalculatorCPU()
{
	return *latticeCalculatorCPU_ ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
const std::string SIMULATION_ENGINE_SPECIALIZATION_BASE::
getLatticeArrangementName() const
{
	return TypeNamesExtractorType::getLatticeArrangementName() ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
const std::string SIMULATION_ENGINE_SPECIALIZATION_BASE::
getFluidModelName() const
{
	return TypeNamesExtractorType::getFluidModelName() ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
const std::string SIMULATION_ENGINE_SPECIALIZATION_BASE::
getCollisionModelName() const
{
	return TypeNamesExtractorType::getCollisionModelName() ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
const std::string SIMULATION_ENGINE_SPECIALIZATION_BASE::
getDataTypeName() const
{
	return TypeNamesExtractorType::getDataTypeName() ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
const std::string SIMULATION_ENGINE_SPECIALIZATION_BASE::
getComputationalEngineName() const
{
	return ComputationalEngine::getName() ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
ComputationError<double> SIMULATION_ENGINE_SPECIALIZATION_BASE::
computeErrorFromCPU( bool shouldSaveVelocityT0 )
{
	ComputationError<double  > result ;
	ComputationError<DataType> localError ;

	if (shouldSaveVelocityT0)
	{
		localError = getLatticeCalculatorCPU().template computeError<true>( tiledLatticeCPU_ ) ;
	}
	else
	{
		localError = getLatticeCalculatorCPU().template computeError<false>( tiledLatticeCPU_ ) ;
	}

	result.error = localError.error ;
	result.maxVelocityLB = localError.maxVelocityLB ;
	result.maxVelocityNodeCoordinates = localError.maxVelocityNodeCoordinates ;

	return result ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
Simulation::NodeLB SIMULATION_ENGINE_SPECIALIZATION_BASE::
getNodeFromCPU( Coordinates coordinates )
{
	Simulation::NodeLB node(coordinates) ;

	//TODO: probably unused, used only for solid nodes, which are not fully compared with RS code.
	node.f        .resize(LatticeArrangement::getQ(),0) ;
	node.fPost    .resize(LatticeArrangement::getQ(),0) ;
	node.u        .resize(LatticeArrangement::getD(),0) ;
	node.uT0      .resize(LatticeArrangement::getD(),0) ;
	node.uBoundary.resize(LatticeArrangement::getD(),0) ;
	node.rho = 0 ;
	node.rhoBoundary = 0 ;


	try //FIXME: very ugly, add method to TiledLattice, which tells if tile is empty or not.
	{
		// below line throws, when coordinates are outside geometry.
		auto tileIndex = tiledLatticeCPU_.getTileLayout().getTile( coordinates ).getIndex() ;
		auto tile = tiledLatticeCPU_.getTile( tileIndex ) ;

		const unsigned tileEdge = tile.getNNodesPerEdge() ; 

		Coordinates nodeInTileCoordinates
									(
										coordinates.getX() % tileEdge,
										coordinates.getY() % tileEdge,
										coordinates.getZ() % tileEdge
									) ;

		auto nodeFromTile = tile.getNode(
																			nodeInTileCoordinates.getX(),
																			nodeInTileCoordinates.getY(),
																			nodeInTileCoordinates.getZ()
																		) ;

		node.coordinates = nodeInTileCoordinates + tile.getCornerPosition() ;
		
		node.baseType = nodeFromTile.nodeType().getBaseType() ;
		node.placementModifier = nodeFromTile.nodeType().getPlacementModifier() ;

		node.rho = nodeFromTile.rho() ;
		node.rhoBoundary = nodeFromTile.rhoBoundary() ;

		for (unsigned i=0 ; i < LatticeArrangement::getD() ; i++)
		{
			DataType u = nodeFromTile.u(i) ;
			node.u[i] = u ;
			node.uBoundary[i] = nodeFromTile.uBoundary(i) ;
			node.uT0[i] = nodeFromTile.uT0(i) ;
		}
		for (Direction::DirectionIndex i=0 ; i < LatticeArrangement::getQ() ; i++)
		{
			node.f[i] = nodeFromTile.f(i) ;
			node.fPost[i] = nodeFromTile.fPost(i) ;
		}
	}
	catch(...)
	{
	}

	return node ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
void SIMULATION_ENGINE_SPECIALIZATION_BASE::
saveCPUToVtkFile( unsigned stepNumber, const Settings & settings )
{
	switch (settings.getVtkDefaultRhoForBB2Nodes())
	{
		case Settings::DefaultValue::MEAN:

			getLatticeCalculatorCPU().computeRhoForBB2Nodes (tiledLatticeCPU_) ;
			break ;


		case Settings::DefaultValue::NOT_A_NUMBER:

			tiledLatticeCPU_.forEachNode
			(
				[&] (typename TiledLatticeTypeCPU::TileType::DefaultNodeType & node,
				 		 Coordinates & globCoord)
				{
					if (node.nodeType() == NodeBaseType::BOUNCE_BACK_2)
					{
						node.rho() = NAN ;
					}
				}
			) ;
			break ;
	}

	std::stringstream filePath ;
	filePath << settings.getOutputDirectoryPath() << "/" ;
	filePath << "microflow_output" << stepNumber ;

	auto writer = Writer <LatticeArrangement,
												DataType,
												tileDataArrangement> (tiledLatticeCPU_) ;
	writer.saveVtk (settings, filePath.str()) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
void SIMULATION_ENGINE_SPECIALIZATION_BASE::
saveCPUtoCheckpoint( unsigned stepNumber, const Settings & settings )
{
	std::stringstream filePath ;
	filePath << settings.getCheckpointDirectoryPath() << "/" ;
	filePath << "microflow_checkpoint" << stepNumber ;

	auto writer = Writer <LatticeArrangement,
												DataType,
												tileDataArrangement> (tiledLatticeCPU_) ;
	writer.saveVtk (CheckpointSettings(), filePath.str()) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
void SIMULATION_ENGINE_SPECIALIZATION_BASE::
loadCPUFromCheckpoint( std::string checkpointFilePath )
{
	auto const ext = getFileExtension (checkpointFilePath) ;

	if (".vti" == ext)
	{
		auto reader = vtkSmartPointer <ReaderVtkImage>::New() ;
		reader->SetFileName (checkpointFilePath.c_str()) ;
		reader->read (tiledLatticeCPU_) ;

		if (reader->GetErrorCode() != vtkErrorCode::NoError)
		{
			THROW("Can not load file " + checkpointFilePath) ;
		}
	}
	else if (".vtu" == ext)
	{
		auto reader = vtkSmartPointer <ReaderVtkUnstructured>::New() ;
		reader->SetFileName (checkpointFilePath.c_str()) ;
		reader->read (tiledLatticeCPU_) ;

		if (reader->GetErrorCode() != vtkErrorCode::NoError)
		{
			THROW("Can not load file " + checkpointFilePath) ;
		}
	}
	else
	{
		THROW ("Can not load checkpoint from \"" + ext + "\" file") ;
	}
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
SIMULATION_ENGINE_SPECIALIZATION::
SimulationEngineSpecialization
( 
	TileLayout<StorageOnCPU> & tileLayout, 
	ExpandedNodeLayout & expandedNodeLayout,
	const Settings & settings 
)
:	SimulationEngineSpecializationBaseType( tileLayout, expandedNodeLayout, settings )
{
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION::
initializeAtEquilibrium()
{
	SimulationEngineSpecializationBaseType::
		getLatticeCalculator().initializeAtEquilibrium( tiledLatticeCPU_ ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION::
collideAndPropagate (bool shouldComputeRhoU __attribute__((unused)))
{
	auto latticeCalculator = SimulationEngineSpecializationBaseType::
													   getLatticeCalculator() ;

	latticeCalculator.collide        ( tiledLatticeCPU_ ) ;
	latticeCalculator.propagate      ( tiledLatticeCPU_ ) ;
	latticeCalculator.processBoundary( tiledLatticeCPU_ ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION::
saveVtk( unsigned stepNumber, const Settings & settings )
{
	SimulationEngineSpecializationBaseType::
		saveCPUToVtkFile( stepNumber, settings ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION::
saveCheckpoint( unsigned stepNumber, const Settings & settings )
{
	SimulationEngineSpecializationBaseType::
		saveCPUtoCheckpoint( stepNumber, settings ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION::
loadCheckpoint( std::string checkpointFilePath )
{
	return SimulationEngineSpecializationBaseType::
		loadCPUFromCheckpoint( checkpointFilePath ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
ComputationError<double> SIMULATION_ENGINE_SPECIALIZATION::
computeError( bool shouldSaveVelocityT0 )
{
	return SimulationEngineSpecializationBaseType::
		computeErrorFromCPU( shouldSaveVelocityT0 ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
SimulationEngine * SIMULATION_ENGINE_SPECIALIZATION::
create
( 
	TileLayout<StorageOnCPU> & tileLayout, 
	ExpandedNodeLayout & expandedNodeLayout,
	const Settings & settings 
)
{
	return new SimulationEngineSpecialization( tileLayout, expandedNodeLayout, settings ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
Simulation::NodeLB SIMULATION_ENGINE_SPECIALIZATION::
getNode( Coordinates coordinates )
{
	return SimulationEngineSpecializationBaseType::getNodeFromCPU( coordinates ) ;
}



#undef SIMULATION_ENGINE_SPECIALIZATION_BASE
#undef TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION_BASE
#undef SIMULATION_ENGINE_SPECIALIZATION
#undef TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION


	
}



#endif
