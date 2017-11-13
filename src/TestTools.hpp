#ifndef TEST_TOOLS_HPP
#define TEST_TOOLS_HPP



#include <cmath>


#ifdef ENABLE_UNSAFE_OPTIMIZATIONS

	#define FPU_CMP(a,b) \
						EXPECT_DOUBLE_EQ( (a), (b) ) << " diff=" << ((a)-(b)) \
									<< ", rel=" << ((a)-(b))/(a) 

	#define FPU_CMP_ASSERT(a,b) \
						if ( not std::isnan((a)) && not std::isnan((b)) ) \
						ASSERT_DOUBLE_EQ( (a), (b) ) << " diff=" << ((a)-(b)) \
									<< ", rel=" << ((a)-(b))/(a) << " "
#else

	#define FPU_CMP(a,b) \
						EXPECT_EQ( (a), (b) ) << " diff=" << ((a)-(b)) \
									<< ", rel=" << ((a)-(b))/(a) 

	#define FPU_CMP_ASSERT(a,b) \
						if ( not std::isnan((a)) && not std::isnan((b)) ) \
						ASSERT_EQ( (a), (b) ) << " diff=" << ((a)-(b)) \
									<< ", rel=" << ((a)-(b))/(a) << " "

#endif





template<class D>
static inline
bool areFloatsNotEqual( D f1, D f2 )
{
	if ( std::isnan(f1)  &&  std::isnan(f2) ) 
	{
		return false ;
	}

#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	return ( abs (f1 - f2) > (1e14 * std::max (abs(f1),abs(f2)))  ) ;
#else
	return ( f1 != f2 ) ;
#endif
}



#endif
