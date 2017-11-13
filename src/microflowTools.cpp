#include "microflowTools.hpp"
#include "Logger.hpp"



#include <sstream>
#include <unistd.h>



using namespace std ;



namespace microflow
{



class SYNTAX_ERROR_EXCEPTION 
{} ;



string bytesToHuman (size_t numberOfBytes)
{
	double divisor = 1 ;
	string unit = "??" ;

	constexpr size_t Ti = 1LL << 40 ;
	constexpr size_t Gi = 1LL << 30 ;
	constexpr size_t Mi = 1LL << 20 ;
	constexpr size_t Ki = 1LL << 10 ;

	if (numberOfBytes > Ti)
	{
		divisor = Ti ;
		unit = "TiB" ;
	}
	else if (numberOfBytes > Gi)
	{
		divisor = Gi ;
		unit = "GiB" ;
	}
	else if (numberOfBytes > Mi)
	{
		divisor = Mi ;
		unit = "MiB" ;
	}
	else if (numberOfBytes > Ki)
	{
		divisor = Ki ;
		unit = "KiB" ;
	}
	else
	{
		divisor = 1 ;
		unit = "B" ;
	}

	stringstream ss ;
	ss << static_cast<long double>(numberOfBytes) / divisor << " " << unit ;

	return ss.str() ;
}



std::string microsecondsToHuman (double microseconds)
{
	double divisor = 1.0 ;
	string unit = "??" ;

	const double s = 1e6 ;
	const double ms = 1e3 ;
	const double us = 1.0 ;

	if (microseconds > s)
	{
		divisor = s ;
		unit = "s" ;
	}
	else if (microseconds > ms)
	{
		divisor = ms ;
		unit = "ms" ;
	}
	else
	{
		divisor = us ;
		unit = "us" ;
	}

	stringstream ss ;
	ss << static_cast<long double>(microseconds) / divisor << " " << unit ;

	return ss.str() ;
}



// Look at:
//		https://bytes.com/topic/c/answers/219178-get-system-memory-c
//		http://ubuntuforums.org/showthread.php?t=1209886
size_t getInstalledPhysicalMemoryInBytes()
{
	size_t numberOfPages = sysconf (_SC_PHYS_PAGES) ;
	size_t pageSize      = sysconf (_SC_PAGE_SIZE)  ;

	return numberOfPages * pageSize ;
}
size_t getFreePhysicalMemoryInBytes()
{
	size_t numberOfPages = sysconf (_SC_AVPHYS_PAGES) ;
	size_t pageSize      = sysconf (_SC_PAGE_SIZE)  ;

	return numberOfPages * pageSize ;
}



void readChar(istream & inputStream, const char requiredChar)
{
	char c ;
	inputStream >> skipws >> c ;

	if ( requiredChar != c )                                      
	{                                                  
		logger << "Format error, got \'" << c << "\' (" << (int)c << ")" ;
		logger << ", expected \'" << requiredChar << "\'\n" ;
		throw SYNTAX_ERROR_EXCEPTION() ;
	}
}



}
