#ifndef SIMULATION_ENGINE_HPP
#define SIMULATION_ENGINE_HPP



#include <string>
#include <memory>



#include "Simulation.hpp"
#include "TileLayout.hpp"
#include "TiledLattice.hpp"
#include "LatticeCalculator.hpp"
#include "TypeNamesExtractor.hpp"
#include "Settings.hpp"
#include "ComputationalEngine.hpp"



namespace microflow
{



class SimulationEngine
{
	public:
		virtual ~SimulationEngine() {} ;

		virtual void initializeAtEquilibrium() = 0 ;
		virtual void collideAndPropagate (bool shouldComputeRhoU) = 0 ;

		virtual void saveVtk( unsigned stepNumber, const Settings & settings ) = 0 ;

		virtual void saveCheckpoint( unsigned stepNumber, const Settings & settings ) = 0 ;
		virtual void loadCheckpoint( std::string checkpointFilePath ) = 0 ;

		virtual ComputationError<double> computeError( bool shouldSaveVelocityT0 = true ) = 0 ;

		virtual const std::string getLatticeArrangementName () const = 0 ;
		virtual const std::string getFluidModelName         () const = 0 ;
		virtual const std::string getCollisionModelName     () const = 0 ;
		virtual const std::string getDataTypeName           () const = 0 ;
		virtual const std::string getComputationalEngineName() const = 0 ;

		virtual typename Simulation::NodeLB getNode( Coordinates coordinates ) = 0 ;
} ;




typedef SimulationEngine * ( * CreateSimulationEngineMethod )
						( 
							TileLayout<StorageOnCPU> &, 
							ExpandedNodeLayout &,
							const Settings &
						) ; 



template<
					class LatticeArrangement,
					template<class, class > class FluidModel,
					class CollisionModel,
					class DataType,
					class ComputationalEngine
				>
class SimulationEngineSpecializationBase
: public SimulationEngine, public TypeNamesExtractor
																			< LatticeArrangement, FluidModel, 
																				CollisionModel, DataType >
{
	public:
		virtual const std::string getLatticeArrangementName () const ;
		virtual const std::string getFluidModelName         () const ;
		virtual const std::string getCollisionModelName     () const ;
		virtual const std::string getDataTypeName           () const ;
		virtual const std::string getComputationalEngineName() const ;

		SimulationEngineSpecializationBase( TileLayout<StorageOnCPU> & tileLayout,
																				ExpandedNodeLayout & expandedNodeLayout, 
																				const Settings & settings ) ;

		virtual ~SimulationEngineSpecializationBase() {} ;


	protected:
		// FIXME: I have problems with using inherited alias template
		//			 	as template template parameter, look at ComputationalEngine.hpp
		//typedef LatticeCalculator
		//				< 
		//					FluidModel, LatticeArrangement, DataType, 
		//					ComputationalEngine::template StorageType
		//				>	 LatticeCalculatorType ;
		static constexpr TileDataArrangement tileDataArrangement = TileDataArrangement::OPT_1 ;

		typedef typename ComputationalEngine::template LatticeCalculatorType
								<FluidModel, CollisionModel, LatticeArrangement, DataType, 
								 tileDataArrangement> LatticeCalculatorType ;
		typedef LatticeCalculator
								<FluidModel, CollisionModel, LatticeArrangement, DataType, 
								 StorageOnCPU, tileDataArrangement> LatticeCalculatorTypeCPU ;
																	
		
		LatticeCalculatorType    & getLatticeCalculator() ;
		LatticeCalculatorTypeCPU & getLatticeCalculatorCPU() ;

		typedef TiledLattice <LatticeArrangement, DataType, StorageOnCPU, 
													tileDataArrangement> TiledLatticeTypeCPU ;

		TiledLatticeTypeCPU tiledLatticeCPU_ ;

		ComputationError<double> computeErrorFromCPU( bool shouldSaveVelocityT0 = true ) ;

		Simulation::NodeLB getNodeFromCPU( Coordinates coordinates ) ;
		void saveCPUToVtkFile( unsigned stepNumber, const Settings & settings ) ;
		void saveCPUtoCheckpoint( unsigned stepNumber, const Settings & settings ) ;
		void loadCPUFromCheckpoint( std::string checkpointFilePath ) ;

	
	private:
		typedef TypeNamesExtractor< LatticeArrangement, FluidModel, CollisionModel, DataType >
							TypeNamesExtractorType ;

		std::unique_ptr<LatticeCalculatorType> latticeCalculator_ ;
		std::unique_ptr<LatticeCalculatorTypeCPU> latticeCalculatorCPU_ ;
} ;



template<
					class LatticeArrangement,
					template<class, class > class FluidModel,
					class CollisionModel,
					class DataType,
					class ComputationalEngine
				>
class SimulationEngineSpecialization  ;



template<
					class LatticeArrangement,
					template<class, class > class FluidModel,
					class CollisionModel,
					class DataType
				>
class SimulationEngineSpecialization
< LatticeArrangement, FluidModel, CollisionModel, DataType, ComputationalEngineCPU > 
: public SimulationEngineSpecializationBase
					< 
						LatticeArrangement, FluidModel, CollisionModel, DataType, 
						ComputationalEngineCPU
					>
{

	public:

		SimulationEngineSpecialization
		( 
			TileLayout<StorageOnCPU> & tileLayout, 
			ExpandedNodeLayout & expandedNodeLayout,
			const Settings & settings 
		) ;
		virtual ~SimulationEngineSpecialization() {} ;


		virtual void initializeAtEquilibrium() ;
		virtual void collideAndPropagate (bool shouldComputeRhoU) ;

		virtual void saveVtk( unsigned stepNumber, const Settings & settings ) ;
		virtual void saveCheckpoint( unsigned stepNumber, const Settings & settings ) ;
		virtual void loadCheckpoint( std::string checkpointFilePath ) ;

		virtual ComputationError<double> computeError( bool shouldSaveVelocityT0 = true ) ;

		// Needed for SimulationEngineFactory
		static SimulationEngine * 
		create
		(
			TileLayout<StorageOnCPU> & tileLayout, 
			ExpandedNodeLayout & expandedNodeLayout,
			const Settings & settings
		) ;

		virtual Simulation::NodeLB getNode( Coordinates coordinates ) ;


	private:

		typedef SimulationEngineSpecializationBase
						<
							LatticeArrangement, FluidModel, CollisionModel, DataType,
							ComputationalEngineCPU
						> SimulationEngineSpecializationBaseType ;

		using SimulationEngineSpecializationBaseType::tiledLatticeCPU_ ;
} ;



template<
					class LatticeArrangement,
					template<class, class > class FluidModel,
					class CollisionModel,
					class DataType
				>
class SimulationEngineSpecialization
< LatticeArrangement, FluidModel, CollisionModel, DataType, ComputationalEngineGPU > 
: public SimulationEngineSpecializationBase
					< 
						LatticeArrangement, FluidModel, CollisionModel, DataType, 
						ComputationalEngineGPU
					>
{

	public:

		SimulationEngineSpecialization
		( 
			TileLayout<StorageOnCPU> & tileLayout, 
			ExpandedNodeLayout & expandedNodeLayout,
			const Settings & settings 
		) ;
		virtual ~SimulationEngineSpecialization() {} ;


		virtual void initializeAtEquilibrium() ;
		virtual void collideAndPropagate (bool shouldComputeRhoU) ;

		virtual void saveVtk( unsigned stepNumber, const Settings & settings ) ;
		virtual void saveCheckpoint( unsigned stepNumber, const Settings & settings ) ;
		virtual void loadCheckpoint( std::string checkpointFilePath ) ;

		virtual ComputationError<double> computeError( bool shouldSaveVelocityT0 = true ) ;

		// Needed for SimulationEngineFactory
		static SimulationEngine * 
		create
		(
			TileLayout<StorageOnCPU> & tileLayout, 
			ExpandedNodeLayout & expandedNodeLayout,
			const Settings & settings
		) ;

		virtual Simulation::NodeLB getNode( Coordinates coordinates ) ;


	private:

		typedef SimulationEngineSpecializationBase
						<
							LatticeArrangement, FluidModel, CollisionModel, DataType,
							ComputationalEngineGPU
						> SimulationEngineSpecializationBaseType ;

		typedef TiledLattice <LatticeArrangement, DataType, StorageOnGPU, 
													TileDataArrangement::OPT_1> TiledLatticeTypeGPU ;

		using SimulationEngineSpecializationBaseType::tiledLatticeCPU_ ;
		TileLayout<StorageOnGPU> tileLayoutGPU_ ;
		TiledLatticeTypeGPU tiledLatticeGPU_ ;

		bool isCPUCopyValid_ ;
		void synchronizeCPUCopy() ;
} ;



// Based on http://www.codeproject.com/Articles/363338/Factory-Pattern-in-Cplusplus
class SimulationEngineFactory
{
	public:

    ~SimulationEngineFactory() ;

		static
    SimulationEngine * createEngine
		( 
			const Settings & settings,
			TileLayout<StorageOnCPU> & tileLayout,
			ExpandedNodeLayout & expandedNodeLayout
		) ;
		static
    SimulationEngine * createEngine
		( 
			std::string engineName,
			const Settings & settings,
			TileLayout<StorageOnCPU> & tileLayout,
			ExpandedNodeLayout & expandedNodeLayout
		) ;


	private:
	
		SimulationEngineFactory() ;
		SimulationEngineFactory(const SimulationEngineFactory &) ;
		SimulationEngineFactory &operator=(const SimulationEngineFactory &) ;

    void registerEngine( std::string engineName, 
												CreateSimulationEngineMethod createMethod ) ;

		static std::string buildEngineName( const Settings & settings ) ;
		static std::string buildEngineName
											 ( 
											 		std::string latticeArrangement,
													std::string fluidModel,
													std::string collisionModel,
													std::string dataType,
													std::string computationalEngine
											 ) ;

		typedef std::map< std::string, CreateSimulationEngineMethod > FactoryMap ;
		FactoryMap factoryMap_ ;
} ;



}



#include "SimulationEngine.hh"
#include "SimulationEngine.tcc"



#endif
