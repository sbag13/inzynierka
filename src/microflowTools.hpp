#ifndef MICROFLOW_TOOLS_HPP
#define MICROFLOW_TOOLS_HPP



#include <istream>
#include <string>



namespace microflow
{



inline
double toPercentage( double value )
{
	return value * 100.0 ;
}



inline
constexpr unsigned sizeInBits(unsigned n, unsigned p = 0) {
    return (n == 0) ? p : sizeInBits(n / 2, p + 1);
}



std::string bytesToHuman (size_t numberOfBytes) ;
std::string microsecondsToHuman (double microseconds) ;

size_t getInstalledPhysicalMemoryInBytes() ;
size_t getFreePhysicalMemoryInBytes() ;



void readChar(std::istream & inputStream, const char requiredChar) ;



}



#endif
