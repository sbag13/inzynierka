#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP



namespace microflow
{



class NoOptimizations
{
	public:
		HD static constexpr bool shouldEnableUnsafeOptimizations() { return false ; }
} ;



class UnsafeOptimizations
{
	public:
		HD static constexpr bool shouldEnableUnsafeOptimizations() { return true ; }
} ;



}



#endif
