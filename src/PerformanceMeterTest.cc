#include "gtest/gtest.h"
#include "PerformanceMeter.hpp"

#include <chrono>
#include <thread>
#include <sstream>



using namespace microflow ;
using namespace std ;




TEST( PerformanceMeter, hasMicrosecondSupport )
{
	EXPECT_TRUE( PerformanceMeter::hasMicrosecondSupport() ) ;
}



TEST( PerformanceMeter, start )
{
	PerformanceMeter pm ;

	EXPECT_NO_THROW( pm.start() ) ;
	EXPECT_ANY_THROW( pm.start() ) ;
}



TEST( PerformanceMeter, start_stop )
{
	PerformanceMeter pm ;

	EXPECT_ANY_THROW( pm.stop() ) ;
	EXPECT_NO_THROW( pm.start() ) ;
	EXPECT_NO_THROW( pm.stop() ) ;
	EXPECT_ANY_THROW( pm.stop() ) ;
	EXPECT_NO_THROW( pm.start() ) ;
}



TEST( PerformanceMeter, getNumberOfMeasures )
{
	PerformanceMeter pm(100) ;

	EXPECT_EQ( 0u, pm.getNumberOfMeasures() ) ;

	pm.start() ; pm.stop() ;
	EXPECT_EQ( 1u, pm.getNumberOfMeasures() ) ;

	pm.start() ; pm.stop() ;
	EXPECT_EQ( 2u, pm.getNumberOfMeasures() ) ;

	pm.start() ; pm.stop() ;
	EXPECT_EQ( 3u, pm.getNumberOfMeasures() ) ;

	pm.start() ; pm.stop() ;
	EXPECT_EQ( 4u, pm.getNumberOfMeasures() ) ;

	for (unsigned i=0 ; i < 1000 ; i++)
	{
		pm.start() ; pm.stop() ;
	}
	EXPECT_EQ( 1004u, pm.getNumberOfMeasures() ) ;

	cout << "min = " << pm.findMinDuration() << " us\n" ;
	cout << "max = " << pm.findMaxDuration() << " us\n" ;
	cout << "avg = " << pm.computeAverageDuration() << " us\n" ;
	cout << "std = " << pm.computeStandardDeviation() << " us\n" ;

	cout << pm.generateSummary() << "\n" ;
}



TEST( PerformanceMeter, generateSummary_units )
{
	PerformanceMeter pm(30) ;

	for (unsigned i=0 ; i < 5 ; i++)
	{
		pm.start() ;
		std::this_thread::sleep_for( std::chrono::microseconds(i) ) ;
		pm.stop() ;
	}


	stringstream ss ;

	ss << pm.generateSummary() ;
	cout << ss.str() << "\n" ;
	EXPECT_EQ( 'u', *(ss.str().rbegin()+1) ) ;


	for (unsigned i=0 ; i < 10 ; i++)
	{
		pm.start() ;
		std::this_thread::sleep_for( std::chrono::milliseconds(i) ) ;
		pm.stop() ;
	}

	ss.str("") ;
	ss << pm.generateSummary() ;
	cout << ss.str() << "\n" ;
	EXPECT_EQ( 'm', *(ss.str().rbegin()+1) ) ;


	{
		PerformanceMeter pm ;

		for (unsigned i=1 ; i < 2 ; i++)
		{
			pm.start() ;
			std::this_thread::sleep_for( std::chrono::seconds(i) ) ;
			pm.stop() ;
		}

		ss.str("") ;
		ss << pm.generateSummary() ;
		cout << ss.str() << "\n" ;
		EXPECT_EQ( ' ', *(ss.str().rbegin()+1) ) ;
	}
}
