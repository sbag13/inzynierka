#ifndef SIMULATION_ENGINE_TCC
#define SIMULATION_ENGINE_TCC



#ifdef __CUDACC__



namespace microflow
{



#define TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION              \
template<                                                      \
					class LatticeArrangement,                            \
					template <class, class> class FluidModel,            \
					class CollisionModel,                                \
					class DataType                                       \
				>



#define SIMULATION_ENGINE_SPECIALIZATION_GPU                   \
SimulationEngineSpecialization                                 \
<                                                              \
	LatticeArrangement, FluidModel, CollisionModel, DataType,    \
	ComputationalEngineGPU                                       \
>



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
SIMULATION_ENGINE_SPECIALIZATION_GPU::
SimulationEngineSpecialization
( 
	TileLayout<StorageOnCPU> & tileLayout, 
	ExpandedNodeLayout & expandedNodeLayout,
	const Settings & settings 
)
:	SimulationEngineSpecializationBaseType( tileLayout, expandedNodeLayout, settings ),
	tileLayoutGPU_( tileLayout ),
	tiledLatticeGPU_( tiledLatticeCPU_, tileLayoutGPU_ ),
	isCPUCopyValid_( true )
{
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION_GPU::
initializeAtEquilibrium()
{
	SimulationEngineSpecializationBaseType::
		getLatticeCalculator().initializeAtEquilibriumForGather (tiledLatticeGPU_) ;

	isCPUCopyValid_ = false ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION_GPU::
collideAndPropagate (bool shouldComputeRhoU)
{
	auto & latticeCalculator = SimulationEngineSpecializationBaseType::
													   getLatticeCalculator() ;

	//FIXME: REMEMBER to modify save methods - in the below call final
	// lattice state is not correct, needed additional propagation and 
	// boundary processing. 

	latticeCalculator.gatherProcessBoundaryCollide 
		(tiledLatticeGPU_, shouldComputeRhoU) ;

	isCPUCopyValid_ = false ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION_GPU::
saveVtk( unsigned stepNumber, const Settings & settings )
{
	synchronizeCPUCopy() ;

	SimulationEngineSpecializationBaseType::
		saveCPUToVtkFile( stepNumber, settings ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION_GPU::
saveCheckpoint( unsigned stepNumber, const Settings & settings )
{
	synchronizeCPUCopy() ;

	SimulationEngineSpecializationBaseType::
		saveCPUtoCheckpoint( stepNumber, settings ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION_GPU::
loadCheckpoint( std::string checkpointFilePath )
{
	SimulationEngineSpecializationBaseType::
		loadCPUFromCheckpoint( checkpointFilePath ) ;

	tiledLatticeGPU_.copyFromCPU( tiledLatticeCPU_ ) ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
ComputationError<double> SIMULATION_ENGINE_SPECIALIZATION_GPU::
computeError( bool shouldSaveVelocityT0 )
{
	synchronizeCPUCopy() ;

	auto result = SimulationEngineSpecializationBaseType::
		computeErrorFromCPU( shouldSaveVelocityT0 ) ;

	//TODO: Since uT0 is not stored on GPU side after error computation, it must be 
	//			preserved in CPU version. Consider either moving error computation to GPU 
	//			or extracting uT0 from the allValues from TiledLattice into separate array.
	tiledLatticeGPU_.copyFromCPU( tiledLatticeCPU_ ) ;

	return result ;
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
SimulationEngine * SIMULATION_ENGINE_SPECIALIZATION_GPU::
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
Simulation::NodeLB SIMULATION_ENGINE_SPECIALIZATION_GPU::
getNode( Coordinates coordinates )
{
	synchronizeCPUCopy() ;

	return SimulationEngineSpecializationBaseType::getNodeFromCPU( coordinates ) ;	
}



TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION
void SIMULATION_ENGINE_SPECIALIZATION_GPU::
synchronizeCPUCopy()
{
	if (not isCPUCopyValid_ )
	{
		tiledLatticeGPU_.copyToCPU( tiledLatticeCPU_ ) ;
		isCPUCopyValid_ = true ;
	}
}



#undef SIMULATION_ENGINE_SPECIALIZATION_GPU
#undef TEMPLATE_SIMULATION_ENGINE_SPECIALIZATION


	
}



#endif //__CUDACC__



#endif
