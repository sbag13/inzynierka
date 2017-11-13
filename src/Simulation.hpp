#ifndef SIMULATION_HPP
#define SIMULATION_HPP



#include <string>
#include <memory> 

#include "Coordinates.hpp"
#include "NodeType.hpp"
#include "Storage.hpp"
#include "Size.hpp"
#include "LatticeCalculator.hpp"
#include "ClassificatorBoundaryAtLocation.hpp"



namespace microflow
{



class SimulationEngine          ;
class Settings                  ;
class ColoredPixelClassificator ;
class Image                     ;
class NodeLayout                ;
class ExpandedNodeLayout        ;
class SimulationEngine          ;
template< template<class> class Storage > class TileLayout ;




class Simulation
{
	public:
		
		Simulation( const std::string casePath ) ;
		~Simulation() ;


		void run() ;
		void stop() ;

		/*
			Computations arranged as collision, propagation, boundary processing.
			Require separate steps in parallel implementation due to synchronization
			problems at propagation.
		*/
		void initializeAtEquilibrium() ;
		// shouldComputeRhoU = false decreases GPU memory transfer and thus 
		// increases performance.
		void collideAndPropagate (bool shouldComputeRhoU = true) ;

		//TODO: I am not sure, where stepNumber should be updated. If it is internal
		//			attribute in Simulation, then every method should CONSISTETLY update
		//			this - seems impossible, unless only Simulation state updates do
		//			full time step computation.
		void saveVtk( unsigned stepNumber ) ;

		void saveCheckpoint( unsigned stepNumber ) ;
		int loadCheckpoint() ;

		// FIXME: probably the argument should be removed, it is unused now
		//        because of problems with robust error computation after
		//				checkpoint load.
		ComputationError<double> computeError( bool shouldSaveVelocityT0 = true ) ;

		Size getSize() const ;


		struct NodeLB
		{
			public:

				NodeLB( Coordinates nodeCoordinates ) ;
				
				NodeBaseType baseType ;
				PlacementModifier placementModifier ;

				std::vector< double > f ;
				std::vector< double > fPost ;
				std::vector< double > u ;
				std::vector< double > uT0 ;
				double rho ;

				std::vector< double > uBoundary ;
				double rhoBoundary ;

				Coordinates coordinates ;
		} ;

		// WARNING: extremely slow method, use only for single nodes.
		// TODO: to keep it const, in GPU specializations there should be some mechanism
		//			 allowing for copying on demand parts of GPU memory instead of 
		//			 full synchronisation with of CPU and GPU copies.
		NodeLB getNode( Coordinates coordinates ) ;

		//FIXME: hack, which allows externally set characteristicLengthLB_
		//			 Remove, when this parameter is automatically extracted from
		//			 geometry (maybe in NodeLayout ?)
		Settings * getSettings() ; 


	private:

		void prepareOutputDirectories( int firstStepNumber ) ;
		void removeRedundantVtkFiles() ;
		void removeRedundantCheckpoints() ;
		void removeRedundantFiles( const std::string directoryPath,
															 const std::string fileNamePattern,
															 unsigned maxNumberOfFiles ) ;
		std::string findCheckpointFile() ;

		Simulation() ;
		Simulation( const Simulation & ) ;
		Simulation & operator=( const Simulation & ) ;

		std::unique_ptr< Settings                  > settings_                  ;
		std::unique_ptr< NodeLayout                > nodeLayout_                ;
		std::unique_ptr< ExpandedNodeLayout        > expandedNodeLayout_        ;
		std::unique_ptr< TileLayout<StorageOnCPU>  > tileLayout_                ;
		std::unique_ptr< SimulationEngine          > simulationEngine_          ;

		std::unique_ptr< ColoredPixelClassificator > coloredPixelClassificator_ ;
		std::unique_ptr< Image                     > image_                     ;

		std::unique_ptr <ClassificatorBoundaryAtLocation> classificatorBoundaryAtLocation_ ;

		bool shouldStop_ ; //TODO: updated from signal handler, avoid race conditions.
} ;



}
#endif
