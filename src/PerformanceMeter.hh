#ifndef PERFORMANCE_METER_HH
#define PERFORMANCE_METER_HH



#include <algorithm>
#include <numeric>

#include "Exceptions.hpp" 



namespace microflow
{



inline
bool PerformanceMeter::
hasMicrosecondSupport()
{
	return Measure::hasMicrosecondSupport() ;
}



inline
PerformanceMeter::
PerformanceMeter( unsigned int initialBufferSize )
{
	measures_.reserve( initialBufferSize ) ;
}



inline
PerformanceMeter::
~PerformanceMeter()
{}



inline
void PerformanceMeter::
start()
{
	currentMeasure_.start() ;
}



inline
void PerformanceMeter::
stop()
{
	currentMeasure_.stop() ;
	measures_.push_back( currentMeasure_ ) ;
	currentMeasure_.reset() ;
}



inline
void PerformanceMeter::
clear()
{
	measures_.clear() ;
	currentMeasure_.reset() ;
}



inline
unsigned PerformanceMeter::
getNumberOfMeasures() const
{
	return measures_.size() ;
}



inline
PerformanceMeter::Measure::
Measure()
{
}



inline
bool PerformanceMeter::Measure::
isInitialized( const TimePoint & timePoint ) 
{
	return ( TimePoint() != timePoint ) ;
}



inline
bool PerformanceMeter::Measure::
hasMicrosecondSupport()
{
	return (
		      1 == Clock::period::num  &&
		1000000 <= Clock::period::den 
	) ;
}



inline
void PerformanceMeter::Measure::
start()
{
	if ( isInitialized( begin ) )
	{
		THROW( "already started" ) ;
	}

	begin = Clock::now() ;
}



inline
void PerformanceMeter::Measure::
stop()
{
	if ( not isInitialized( begin ) )
	{
		THROW( "not started" ) ;
	}
	if ( isInitialized(end) )
	{
		THROW( "already stopped" ) ;
	}

	end = Clock::now() ;
}



inline
void PerformanceMeter::Measure::
reset()
{
	begin = TimePoint() ;
	end = TimePoint() ;
}



inline
PerformanceMeter::Measure::Clock::duration PerformanceMeter::Measure::
computeDurationInMicroseconds() const
{
	return std::chrono::duration_cast<std::chrono::microseconds>( end - begin ) ;
}



}



#endif
