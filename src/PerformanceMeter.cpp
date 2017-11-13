#include "PerformanceMeter.hpp"

#include <sstream>



namespace microflow
{



unsigned long long PerformanceMeter::
findMinDuration()
{
	using namespace std::chrono ;

	unsigned long long result = std::numeric_limits<unsigned long long>::max() ;
	
	for (auto measure : measures_ )
	{
		auto duration = measure.computeDurationInMicroseconds() ;

		result = std::min( result, 
									static_cast<unsigned long long>(duration.count()) ) ;
	}

	return result ;
}



unsigned long long PerformanceMeter::
findMaxDuration()
{
	using namespace std::chrono ;

	unsigned long long result = std::numeric_limits<unsigned long long>::min() ;
	
	for (auto measure : measures_ )
	{
		auto duration = measure.computeDurationInMicroseconds() ;

		result = std::max( result, 
									static_cast<unsigned long long>(duration.count()) ) ;
	}

	return result ;
}



double PerformanceMeter::
computeAverageDuration()
{
	if (0 == getNumberOfMeasures() )
	{
		return 0 ;
	}

	unsigned long long sum = 0 ;

	for (auto measure : measures_ )
	{
		auto duration = measure.computeDurationInMicroseconds() ;
		sum += duration.count() ;
	}

	return static_cast<double>(sum) / getNumberOfMeasures() ;
}



double PerformanceMeter::
computeStandardDeviation()
{
	if ( 0 == getNumberOfMeasures() )
	{
		return 0 ;
	}

  double avg = computeAverageDuration();
  double accum = 0 ;

	for( auto measure : measures_ )
	{
		auto duration = measure.computeDurationInMicroseconds() ;
		double v = duration.count() ;
		accum += (v - avg) * (v - avg) ;
	}

  return sqrt( accum / getNumberOfMeasures() );  
}



std::string PerformanceMeter::
generateSummary()
{
	std::stringstream ss ;

	ss << "measured " << getNumberOfMeasures() << " times" ;
  
	double avg = computeAverageDuration() ;

  double multiplier = 0.0 ;
  std::string unit = " ??" ;
  if ( avg > 1e6 )
  {
    multiplier = 1e-6 ;
    unit = " s" ;
  }  
  else if ( avg > 1e3 ) 
  {
    multiplier = 1e-3 ;
    unit = " ms" ;
  }  
  else 
  {
    multiplier = 1 ;
    unit = " us" ;
  }

  ss << ", on average "    << avg * multiplier << unit ;
	ss << " per one run" ;
  ss << ", std deviation " << computeStandardDeviation() * multiplier << unit ;
  ss << ", the shortest time = " << findMinDuration() * multiplier << unit ;
  ss << ", the longest time = "  << findMaxDuration() * multiplier << unit ;	

	return ss.str() ;
}



}
