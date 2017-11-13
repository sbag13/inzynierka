#include "Simulation.hpp"
#include "Settings.hpp"
#include "CollisionModels.hpp"
#include "FluidModels.hpp"
#include "TiledLattice.hpp"
#include "SimulationEngine.hpp"
#include "ExpandedNodeLayout.hpp"
#include "Logger.hpp"
#include "PerformanceMeter.hpp"
#include "TilingStatistic.hpp"
#include "fileUtils.hpp"
#include "ReaderVtk.hpp"



using namespace std ;



namespace microflow
{



Simulation::
Simulation( const std::string casePath )
{
	logger << "Reading simulation from " << casePath << "\n" ;


	settings_.reset( new Settings( casePath ) ) ;


	if (settings_->isGeometryDefinedByPng())
	{
		settings_->setGeometryOrigin (UniversalCoordinates<double> (0,0,0)) ;

		coloredPixelClassificator_.reset
		( 
			new ColoredPixelClassificator 
						(settings_->getPixelColorDefinitionsFilePath())
		) ;

		image_.reset
		( 
			new Image (settings_->getGeometryPngImagePath()) 
		) ;

		nodeLayout_.reset
		( 
			new NodeLayout (*coloredPixelClassificator_, *image_, 
											settings_->getZExpandDepth()) 
		) ;

	}
	else if (settings_->isGeometryDefinedByVti())
	{

		nodeLayout_.reset
		( 
			new NodeLayout (Size())
		) ;

		auto reader = vtkSmartPointer <ReaderVtkImage>::New() ;
		reader->SetFileName (settings_->getGeometryVtiImagePath().c_str()) ;
		reader->readNodeLayout (*nodeLayout_) ;

		if (std::isnan(reader->getPhysicalSpacing()))
		{
			logger << "WARNING: No \"PhysicalSpacing\" in geometry file, assuming "
						 << "computed physical lattice spacing (" 
						 << settings_->getLatticeSpacingPhysical() << ").\n" ;
		}
		if (not std::isnan(reader->getPhysicalSpacing()) &&
				reader->getPhysicalSpacing() != settings_->getLatticeSpacingPhysical())
		{
			logger << "WARNING: Computed physical lattice spacing (" 
						 << settings_->getLatticeSpacingPhysical()
				     << ") differs from \"PhysicalSpacing\" in geometry file ("
				     << reader->getPhysicalSpacing() << ").\n"
						 << "         Difference is " 
						 << abs(settings_->getLatticeSpacingPhysical() - 
						 				reader->getPhysicalSpacing()) << ".\n" ;
		}
		auto physicalOrigin = reader->getPhysicalOrigin() ;
		if (std::isnan (physicalOrigin.getX()) ||
				std::isnan (physicalOrigin.getY()) ||
				std::isnan (physicalOrigin.getZ()))
		{
			physicalOrigin = UniversalCoordinates<double> (0,0,0) ;
			logger << "WARNING: No \"PhysicalOrigin\" in geometry file, assuming "
						 << physicalOrigin << ".\n" ;
		}
		else
		{
			logger << "Found \"PhysicalOrigin\" in geometry file: " 
						 << physicalOrigin << ".\n" ;
		}
		settings_->setGeometryOrigin (physicalOrigin) ;


		classificatorBoundaryAtLocation_.reset
		(
			new ClassificatorBoundaryAtLocation (settings_->getGeometryDirectoryPath())
		) ;

		nodeLayout_->setBoundaryDefinitions 
			(classificatorBoundaryAtLocation_->getBoundaryDefinitions()) ;

		classificatorBoundaryAtLocation_->setBoundaryNodes (*nodeLayout_) ;
	}


	settings_->initialModify (*nodeLayout_) ;

	expandedNodeLayout_.reset
	(
		new ExpandedNodeLayout( *nodeLayout_ )
	) ;
	//FIXME: ExpandedNodeLayout::rebuildBoundaryNodes() hangs, when Size::z = 1
	if (1 < settings_->getZExpandDepth() || settings_->isGeometryDefinedByVti())
	{
		expandedNodeLayout_->computeSolidNeighborMasks() ;
		expandedNodeLayout_->computeNormalVectors() ;
		nodeLayout_->temporaryMarkUndefinedBoundaryNodesAndCovers() ;
		if (settings_->isGeometryDefinedByPng())
		{
			nodeLayout_->restoreBoundaryNodes( *coloredPixelClassificator_, *image_ ) ;
		}
		else if (settings_->isGeometryDefinedByVti())
		{
			classificatorBoundaryAtLocation_->setBoundaryNodes (*nodeLayout_) ;
		}
		expandedNodeLayout_->classifyNodesPlacedOnBoundary (*settings_) ;
		expandedNodeLayout_->classifyPlacementForBoundaryNodes (*settings_) ;
	}

	settings_->finalModify (*nodeLayout_) ;
	//TODO: unoptimal (computed twice), but easy.
	expandedNodeLayout_->computeSolidNeighborMasks() ;
	expandedNodeLayout_->computeNormalVectors() ;
	//FIXME: add code for setting of uB and rhoB from Ruby script(s) finalGeometryModificator.rb


	// FIXME: reload settings after obtaining geometry size and characteristic length !!!


	tileLayout_.reset
	( 
		new TileLayout<StorageOnCPU>( *nodeLayout_ ) 
	) ;

	simulationEngine_.reset
	( 
		SimulationEngineFactory::createEngine( *settings_, *tileLayout_, *expandedNodeLayout_ ) 
	) ;

	shouldStop_ = false ;
}



Simulation::
~Simulation()
{
}



void Simulation::
run()
{
	logger << "\nSetting simulation.\n\n" ;


	unsigned stepNumber ;
	ComputationError<double> computationError ;
	computationError.error = 2.0 ;
	
	int checkpointNumber = loadCheckpoint() ;

	prepareOutputDirectories( checkpointNumber ) ;

	if (0 <= checkpointNumber )
	{
		stepNumber = checkpointNumber ;

		logger << "Checkpoint loaded, starting from step number = " << stepNumber
					 << " (" << stepNumber * settings_->getLatticeTimeStepPhysical() << " [s]).\n" ;
	}
	else
	{
		stepNumber = 0 ;
		simulationEngine_->initializeAtEquilibrium() ;

		logger << "No checkpoint found in " << settings_->getCheckpointDirectoryPath()
					 << ", starting from t=0.\n" ;
	}


	logger << "\nStarting simulation.\n\n" ;


	bool vtkFileSaved = false ;
	bool checkpointSaved = false ;

	if (0 != settings_->getMaxNumberOfVtkFiles())
	{
		saveVtk( stepNumber ) ;
		vtkFileSaved = true ;
	}


	PerformanceMeter kernelPerformanceMeterFast (100000) ; // arbitrary val
	PerformanceMeter kernelPerformanceMeterSlow (1000)   ; // arbitrary val

	while ( not shouldStop_  &&
					computationError.error > settings_->getRequiredVelocityRelativeError() )
	{
		stepNumber ++ ; //Increased IN ADVANCE to allow shouldComputeRhoU.

		bool shouldComputeRhoU = 
			settings_->shouldSaveVtkInThisStep        (stepNumber) ||
			settings_->shouldComputeErrorInThisStep   (stepNumber) ||
			settings_->shouldSaveCheckpointInThisStep (stepNumber) ;

		if (shouldComputeRhoU)
		{
			kernelPerformanceMeterSlow.start() ;
				collideAndPropagate (shouldComputeRhoU) ;
			kernelPerformanceMeterSlow.stop() ;
		}
		else
		{
			kernelPerformanceMeterFast.start() ;
				collideAndPropagate (shouldComputeRhoU) ;
			kernelPerformanceMeterFast.stop() ;
		}

		vtkFileSaved = false ;
		checkpointSaved = false ;

		if ( settings_->shouldSaveVtkInThisStep( stepNumber ) )
		{
			saveVtk( stepNumber ) ;
			removeRedundantVtkFiles() ;
			vtkFileSaved = true ;
		}
		
		if ( settings_->shouldComputeErrorInThisStep( stepNumber ) )
		{
			computationError = computeError() ;
			logger << "Time step " << stepNumber 
						 << " (" << stepNumber * settings_->getLatticeTimeStepPhysical() << " [s]): "
						 << computationError << "\n" ;
		}

		if ( settings_->shouldSaveCheckpointInThisStep( stepNumber ) )
		{
			saveCheckpoint( stepNumber ) ;
			removeRedundantCheckpoints() ;
			checkpointSaved = true ;
		}
	}

	if ( shouldStop_ )
	{
		logger << "\nSimulation stopped at step " << stepNumber << "\n\n" ;
	}


	if ( false == vtkFileSaved  &&  0 != settings_->getMaxNumberOfVtkFiles() )
	{
		saveVtk( stepNumber ) ;
		removeRedundantVtkFiles() ;
	}

	if ( false == checkpointSaved  &&  0 != settings_->getMaxNumberOfCheckpoints() )
	{
		saveCheckpoint( stepNumber ) ;
		removeRedundantCheckpoints() ;
	}

	//stepNumber -- ; //FIXME: discuss with RS


	logger << "Simulation finished.\n" ;
	logger << "Final step number = " << stepNumber ;
	//logger << " (" << stepNumber+1 << " steps computed)" //FIXME: discuss with RS
	logger << "\n" ; 
	logger << "Physical time corresponding to final simulation step = " ;
	logger << stepNumber * settings_->getLatticeTimeStepPhysical() ;
	logger << " [s]\n" ;

	logger << "\n" ;

	if (0 == kernelPerformanceMeterFast.getNumberOfMeasures() &&
			0 == kernelPerformanceMeterSlow.getNumberOfMeasures())
	{
		return ;
	}

	unsigned int numberOfNonsolidNodes = 
					tileLayout_->computeTilingStatistic().getNNonSolidNodes() ;

	logger << "\nLBM kernels: "
				 << "per iteration processed "  << numberOfNonsolidNodes 
				 << " non-solid nodes" ;

	if (0 < kernelPerformanceMeterFast.getNumberOfMeasures())
	{
		logger << ",\n" ;
		logger << "fast kernel: " << kernelPerformanceMeterFast.generateSummary()
					 << ", kernel peak performance : "
					 << fixed << (double(numberOfNonsolidNodes) /
					 		kernelPerformanceMeterFast.findMinDuration())
					 << " MLUPS" ;
	}
	if (0 < kernelPerformanceMeterSlow.getNumberOfMeasures())
	{
		logger << ",\n" ;
		logger << "slow kernel: " << kernelPerformanceMeterSlow.generateSummary()
					 << ", kernel peak performance : "
					 << fixed << (double(numberOfNonsolidNodes) /
												kernelPerformanceMeterSlow.findMinDuration())
					 << " MLUPS" ;
	}

	logger << ".\n" ;

}



void Simulation::
stop()
{
	//TODO: this method is called asynchronically from signal handler
	// To avoid race conditions DO NOT update shouldStop_ in other places !
	shouldStop_ = true ;
}



void Simulation::
initializeAtEquilibrium()
{
	simulationEngine_->initializeAtEquilibrium() ;
}



void Simulation::
collideAndPropagate (bool shouldComputeRhoU)
{
	simulationEngine_->collideAndPropagate (shouldComputeRhoU) ;
}



void Simulation::
saveVtk( unsigned stepNumber )
{
	simulationEngine_->saveVtk( stepNumber, *settings_ ) ;
}



void Simulation::
saveCheckpoint( unsigned stepNumber )
{
	simulationEngine_->saveCheckpoint( stepNumber, *settings_ ) ;
}



int Simulation::
loadCheckpoint()
{
	string checkpointFile = findCheckpointFile() ;

	if ("" != checkpointFile)
	{
		logger << "Loading checkpoint from " << checkpointFile << "\n" ;

		simulationEngine_->loadCheckpoint( settings_->getCheckpointDirectoryPath() + 
																									checkpointFile ) ;

		// FIXME: extract from VTK file (add field TimeStep ? )
		return extractStepNumberFromVtkFileName( checkpointFile, "microflow_checkpoint" ) ;
	}

	return -1 ;
}



ComputationError<double> Simulation::
computeError( bool shouldSaveVelocityT0 )
{
	return simulationEngine_->computeError( shouldSaveVelocityT0 ) ;
}



Simulation::NodeLB::
NodeLB( Coordinates nodeCoordinates )
{
	coordinates = nodeCoordinates ;
	baseType = NodeBaseType::SOLID ;
	placementModifier = PlacementModifier::SIZE ;

	f.resize(0) ;
	fPost.resize(0) ;
	u.resize(0) ;
	uT0.resize(0) ;
	rho = NAN ;
	uBoundary.resize(0) ;
	rhoBoundary = NAN ;
}



Simulation::NodeLB Simulation::
getNode( Coordinates coordinates ) 
{
	return simulationEngine_->getNode( coordinates ) ;
}



Size Simulation::
getSize() const
{
	return tileLayout_->getNodeLayout().getSize() ;
}



Settings * Simulation::
getSettings()
{
	return  &(*settings_) ;
}



void Simulation::
prepareOutputDirectories(int firstStepNumber)
{
	if (firstStepNumber < 0) 
	{
		firstStepNumber = 0 ;
	}

	string outputDirectoryPath = settings_->getOutputDirectoryPath() ;

	createDirectory (outputDirectoryPath) ;
	createDirectory (settings_->getCheckpointDirectoryPath()) ;

	logger << "Cleaning directory " << outputDirectoryPath << "\n" ;

	vector<string> file_names = getFileNamesFromDirectory (outputDirectoryPath) ;

	sort( file_names.begin(), file_names.end() ) ;

	for (size_t i=0 ; i < file_names.size() ; i++)
	{
		string d_name = file_names[ i ]  ;

		logger << d_name << " : " ;
		if ( "microflow_param.csv" == d_name     ||
				extractStepNumberFromVtkFileName (d_name, "microflow_output") >=
				static_cast<long long int> (firstStepNumber) 
			 )
		{
			logger << "removing\n" ;
			removeDirectory (outputDirectoryPath + d_name) ;
		} else {
			logger << "leaving\n" ;
		}
	}	
}



void Simulation::
removeRedundantVtkFiles()
{
	removeRedundantFiles
	( 
		settings_->getOutputDirectoryPath(),
		"microflow_output",
		settings_->getMaxNumberOfVtkFiles()
	) ;
}



void Simulation::
removeRedundantCheckpoints()
{
	removeRedundantFiles
	( 
		settings_->getCheckpointDirectoryPath(),
		"microflow_checkpoint",
		settings_->getMaxNumberOfCheckpoints()
	) ;
}



void Simulation::
removeRedundantFiles( const string directoryPath,
											const string fileNamePattern,
											unsigned maxNumberOfFiles )
{
	if ( 0 == maxNumberOfFiles ) return ;


	struct ValidFile
	{
		ValidFile (int stepNr, string name)
			: stepNumber (stepNr), fileName (name)
		{}

		int stepNumber ;
		string fileName ;
	} ;

	vector <ValidFile> validFiles ;


	vector<string> fileNames = getFileNamesFromDirectory( directoryPath ) ;

	for (auto file : fileNames)
	{
		int stepNumber = extractStepNumberFromVtkFileName (file, fileNamePattern) ;
		if (stepNumber > 0)
		{
			validFiles.push_back (ValidFile (stepNumber,file)) ;
		}
	}

	if (validFiles.size() > maxNumberOfFiles)
	{
		sort (validFiles.begin(), validFiles.end(),

					[] (ValidFile const & vf1, ValidFile const & vf2)
					{
						return vf1.stepNumber < vf2.stepNumber ;
					}
		) ;

		for (size_t i=0 ; i < (validFiles.size() - maxNumberOfFiles) ; i++)
		{
			string filePath = directoryPath + validFiles [i].fileName ;
			removeDirectory (filePath) ;
			logger << "Removed " << filePath << "\n" ;
		}
	}	
}



string Simulation::
findCheckpointFile()
{
	vector<string> fileNames = getFileNamesFromDirectory
															( settings_->getCheckpointDirectoryPath() ) ;

	string checkpointFile = "" ;
	int lastCheckpointStepNumber = -1 ;
	for ( auto file : fileNames )
	{
		int checkpointStepNumber = 
						extractStepNumberFromVtkFileName( file, "microflow_checkpoint" ) ;

		if ( checkpointStepNumber > lastCheckpointStepNumber )
		{
			lastCheckpointStepNumber = checkpointStepNumber ;
			checkpointFile = file ;
		}
	}

	return checkpointFile ;
}



}
