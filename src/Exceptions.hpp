#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP



#include <cassert>

#include "Logger.hpp"
#include "cudaPrefix.hpp"



namespace microflow
{



class SYNTAX_ERROR_EXCEPTION 
{} ;



#ifdef NDEBUG

	//FIXME: due to lack of assertions some tests fail after compilation with -DNDEBUG
	#define ASSERT( assertion )

#else

	#ifdef __CUDA_ARCH__

		#define ASSERT( assertion ) \
		if (false == assertion)     \
		{                           \
			asm("trap;") ;            \
		} ;

	#else

		#define ASSERT( assertion ) \
			assert (assertion) ;

	#endif

#endif



inline
HD void
throwException( const char * str, const char * file = __FILE__, int line = __LINE__ )
{
#ifdef __CUDA_ARCH__
	ASSERT (false) ;
#else
	logger << "\nEXCEPTION in " << file << ":" << line << ": " << str << "\n" ;
	throw str ;
#endif	
}



inline void
throwException( const std::string & str, const char * file = __FILE__, int line = __LINE__ )
{
	throwException( str.c_str(), file, line ) ;
}



#define THROW( exceptionMesage ) \
	microflow::throwException( exceptionMesage, __FILE__, __LINE__ ) ;



}
#endif
