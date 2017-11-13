#include "Simulation.hpp"
#include "Logger.hpp"
#include "ProgramParameters.hpp"

#include <signal.h>



microflow::Simulation * simulationPtr = NULL ;



void handleSigInt(int s)
{
	s+=1 ; // avoid compiler warning
	if (NULL == ::simulationPtr)
	{
		THROW("Ctrl-C pressed while simulation uninitialized.\n") ; 
	} else {
		::simulationPtr->stop() ;
	}
}



void registerSigIntHandler()
{
	struct sigaction sigIntHandler;

	sigIntHandler.sa_handler = handleSigInt ;
	sigemptyset(&sigIntHandler.sa_mask) ;
	sigIntHandler.sa_flags = 0 ;

	sigaction(SIGINT, &sigIntHandler, NULL);
} ;



int main(int argc, char ** argv)
{
	microflow::ProgramParameters programParameters(argc, argv) ;

	if (not programParameters.canContinue() )
	{
		return 0 ;
	}

	microflow::initializeGPU( programParameters.getGpuId() ) ;
	{
		microflow::Simulation simulation( programParameters.getCasePath() ) ;

		simulationPtr = & simulation ;
		registerSigIntHandler() ;


		simulation.run() ;

	}
	//http://stackoverflow.com/questions/11608350/proper-use-of-cudadevicereset
	microflow::finalize() ;


	return 0 ;
}
