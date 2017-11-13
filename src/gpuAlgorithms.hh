#ifndef GPU_ALGORITHMS_HH
#define GPU_ALGORITHMS_HH



#include "cudaPrefix.hpp"



namespace microflow
{



inline
HD
void swap (double & val1, double & val2)
{
	double tmp ;
	
	tmp = val1 ;
	val1 = val2 ;
	val2 = tmp ;
}


	
}



#endif
