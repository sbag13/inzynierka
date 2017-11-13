#include "SimulationEngine.hpp"
#include "FluidModels.hpp"
#include "CollisionModels.hpp"
#include "LatticeArrangement.hpp"
#include "Exceptions.hpp"


#include <cctype>



namespace microflow
{



SimulationEngineFactory::
SimulationEngineFactory()
{
#define REGISTER_ENGINE(                                                         \
				latticeArrangement, fluidModel, collisionModel, dataType,                \
				computationalEngine                                                      \
		)                                                                            \
	registerEngine                                                                 \
	(                                                                              \
		buildEngineName                                                              \
		(                                                                            \
			#latticeArrangement, #fluidModel, #collisionModel, #dataType,              \
			#computationalEngine                                                       \
		),                                                                           \
		& SimulationEngineSpecialization                                             \
			< latticeArrangement, fluidModel, collisionModel, dataType,                \
				computationalEngine >::create                                            \
	) ;


	REGISTER_ENGINE
	( 
		D3Q19, FluidModelIncompressible, CollisionModelBGK, double, ComputationalEngineCPU 
	) ;
	REGISTER_ENGINE
	( 
		D3Q19, FluidModelIncompressible, CollisionModelBGK, double, ComputationalEngineGPU
	) ;
	REGISTER_ENGINE
	( 
		D3Q19, FluidModelIncompressible, CollisionModelMRT, double, ComputationalEngineCPU 
	) ;
	REGISTER_ENGINE
	( 
		D3Q19, FluidModelIncompressible, CollisionModelMRT, double, ComputationalEngineGPU
	) ;
	REGISTER_ENGINE
	( 
		D3Q19, FluidModelQuasicompressible, CollisionModelBGK, double, ComputationalEngineCPU
	) ;
	REGISTER_ENGINE
	( 
		D3Q19, FluidModelQuasicompressible, CollisionModelBGK, double, ComputationalEngineGPU
	) ;
	REGISTER_ENGINE
	( 
		D3Q19, FluidModelQuasicompressible, CollisionModelMRT, double, ComputationalEngineCPU
	) ;
	REGISTER_ENGINE
	( 
		D3Q19, FluidModelQuasicompressible, CollisionModelMRT, double, ComputationalEngineGPU
	) ;


#undef REGISTER_ENGINE
}



SimulationEngineFactory::
~SimulationEngineFactory()
{
	factoryMap_.clear() ;
}



void SimulationEngineFactory::
registerEngine( std::string engineName, CreateSimulationEngineMethod createMethod)
{
	factoryMap_[ engineName ] = createMethod ;
}



SimulationEngine * SimulationEngineFactory::
createEngine
( 
	const Settings & settings, 
	TileLayout<StorageOnCPU> & tileLayout,
	ExpandedNodeLayout & expandedNodeLayout
)
{
	return createEngine
					(
						buildEngineName( settings ), settings, tileLayout, expandedNodeLayout
					) ;
}



SimulationEngine * SimulationEngineFactory::
createEngine
( 
	std::string engineName, const 
	Settings & settings, 
	TileLayout<StorageOnCPU> & tileLayout,
	ExpandedNodeLayout & expandedNodeLayout
)
{
	static SimulationEngineFactory instance;

	FactoryMap::iterator it = instance.factoryMap_.find( engineName );
	if( it != instance.factoryMap_.end() )
	{
		return it->second( tileLayout, expandedNodeLayout, settings );
	}

	THROW( std::string("Not found SimulationEngine for ") + engineName ) ;

	return NULL ; //avoid compiler warning, unused because of earlier exception.
}



std::string SimulationEngineFactory::
buildEngineName( const Settings & settings )
{
	return settings.getLatticeArrangementName() + "_" +
				 settings.getFluidModelName()         + "_" +
				 settings.getCollisionModelName()     + "_" +
				 settings.getDataTypeName()           + "_" +
				 settings.getComputationalEngineName() ;
}


	
std::string SimulationEngineFactory::
buildEngineName
(
 std::string latticeArrangement,
 std::string fluidModel,
 std::string collisionModel,
 std::string dataType,
 std::string computationalEngine
 )
{
	std::string name = latticeArrangement + "_" ;

	// TODO: the same names are duplicated in Ruby scripts, add generic mechanism.
	// 			 Maybe short and full names in FluidModel classes ?
	if ( "FluidModelIncompressible" == fluidModel )
	{
		name += "incompressible" ;
	} 
	else if ( "FluidModelQuasicompressible" == fluidModel )
	{
		name += "quasi_compressible" ;
	}
	else
	{
		THROW (std::string("Unknown fluid model specifier: ") + fluidModel ) ;
	}

	name += "_" ;

	// TODO: names duplicated in Ruby scripts, add generic mechanism.
	if ( "CollisionModelBGK" == collisionModel )
	{
		name += "BGK" ;
	}
	else if ( "CollisionModelMRT" == collisionModel )
	{
		name += "MRT" ;
	}
	else
	{
		THROW (std::string("Unknown collision model specifier: ") + collisionModel ) ;
	}

	name += "_" + dataType + "_" ;

	//TODO: duplicated names in Ruby
	if ( "ComputationalEngineCPU" == computationalEngine )
	{
		name += "CPU" ;
	}
	else if ( "ComputationalEngineGPU" == computationalEngine )
	{
		name += "GPU" ;
	}
	else
	{
		THROW (std::string("Unknown computational engine specifier: ") + computationalEngine ) ;
	}

	return name ;
}


	
}
