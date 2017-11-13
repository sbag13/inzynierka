#include "gtest/gtest.h"



#include "Storage.hpp"



using namespace microflow ;



template< template <class T1> class Storage1, 
					template <class T1> class Storage2,
					class T > 
void testStorageCopy()
{
	constexpr unsigned s = 100 ;
	Storage1<T> vcpu1 ; vcpu1.resize(s) ;
	Storage2<T> vcpu2 ; vcpu2.resize(s+5) ;

	for (unsigned i=0 ; i < s ; i++)
	{
		vcpu1[i] = i ;
		vcpu2[i] = i + 10 ;
	}

	EXPECT_FALSE( vcpu1 == vcpu2 ) ;

	vcpu1 = vcpu2 ;

	EXPECT_TRUE( vcpu1 == vcpu2 ) ;
}



TEST( Storage, copyCPUtoCPU )
{
	testStorageCopy< StorageOnCPU, StorageOnCPU, int >() ;
}



TEST( Storage, copyGPUtoCPU )
{
	testStorageCopy< StorageOnCPU, StorageOnGPU, int >() ;
}



TEST( Storage, copyGPUtoGPU )
{
	testStorageCopy< StorageOnGPU, StorageOnGPU, int >() ;
}



TEST( Storage, copyCPUtoGPU )
{
	testStorageCopy< StorageOnGPU, StorageOnCPU, int >() ;
}
