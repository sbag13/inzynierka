#ifndef PROGRAM_PARAMATERS_HPP
#define PROGRAM_PARAMATERS_HPP



#include <string>



namespace microflow
{



class ProgramParameters
{
	public:

		ProgramParameters(int argc, char** argv) ;

		unsigned int getGpuId() ;
		std::string getCasePath() ;
		bool canContinue() ;


	private:

		unsigned int gpuId_ ;
		std::string casePath_ ;
		bool canContinue_ ;
} ;



}
#endif
