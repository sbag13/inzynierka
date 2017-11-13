#include "ProgramParameters.hpp"
#include "Logger.hpp"

#include <getopt.h>
#include <iostream>




namespace microflow
{



ProgramParameters::
ProgramParameters(int argc, char** argv) :
	gpuId_(0),
	casePath_("./"),
	canContinue_(true)
{
	while (true)
	{
		static struct option long_options[] =
		{
			{"gpuIndex", required_argument, 0, 'g' },
			{"casePath", required_argument, 0, 'c' },
			{"help"    , no_argument      , 0, 'h' },
			{0, 0, 0, 0}
		};

		char c = getopt_long (argc, argv, "g:c:h", long_options, NULL);

		/* Detect the end of the options. */
		if (c == -1)
			break;

		switch (c)
		{
			case 'g':
			{
				int g = atoi(optarg) ; 
				if (g >= 0)
				{
					gpuId_ = g ;
				} 
				else {
					logger << "WARNING: GPU index can not be negative, assuming 0\n" ;
				}
			}
			break;

			case 'c':
			{
				casePath_ = optarg ;
			}
			break ;

			case 'h':
			{
				canContinue_ = false ;
				logger << "\nUsage: microflow [-g gpuIndex] [-c casePath] [-h]\n\n" ;
			}
			break ;

			default:
				abort() ;
		}		
	}

	if (optind < argc) 
	{
		canContinue_ = false ;
		logger << "\nUnknown parameters passed, usage: "
							"microflow [-g gpuIndex] [-c casePath] [-h]\n\n" ;
	}
}



unsigned int ProgramParameters::
getGpuId()
{
	return gpuId_ ;
}



std::string ProgramParameters::
getCasePath()
{
	return casePath_ ;
}



bool ProgramParameters::
canContinue()
{
	return canContinue_ ;
}



}
