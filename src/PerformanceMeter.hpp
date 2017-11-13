#ifndef PERFORMANCE_METER_HPP
#define PERFORMANCE_METER_HPP



#include <vector>
#include <string>

#include <chrono>
//http://www.guyrutenberg.com/2013/01/27/using-stdchronohigh_resolution_clock-example/


namespace microflow
{


// TODO: measures only wall clock.
class PerformanceMeter
{
  public:

		static bool hasMicrosecondSupport() ;

    PerformanceMeter(unsigned int initialBufferSize=0) ;
    ~PerformanceMeter() ;

    void start() ;
    void stop () ;

		void clear() ;

    unsigned getNumberOfMeasures () const ;

		// All values in microseconds.
		// FIXME: below methods are UNTESTED due to inexact duration values during measurements.
    unsigned long long findMinDuration() ;
    unsigned long long findMaxDuration() ;
    double computeAverageDuration() ;
    double computeStandardDeviation() ;

		std::string generateSummary() ;


  private:

		class Measure
		{
			public:

				typedef std::chrono::high_resolution_clock Clock ;
				typedef Clock::time_point TimePoint ;

				Measure() ;

				void start() ;
				void stop() ;
				void reset() ;
				Clock::duration computeDurationInMicroseconds() const ;

				static bool isInitialized( const TimePoint & timePoint ) ;
				static bool hasMicrosecondSupport() ;

				TimePoint begin ;
				TimePoint end ;
				
		} ;
    
    std::vector<Measure> measures_ ;
    Measure currentMeasure_ ;
} ;



}



#include "PerformanceMeter.hh"



#endif
