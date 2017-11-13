#ifndef CALCULATOR_HH
#define CALCULATOR_HH



namespace microflow
{



template< 
					template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class LatticeArrangement, 
														class DataType,
					template<class T> class Storage >
Calculator<FluidModel, LatticeArrangement, DataType, Storage >::
Calculator(DataType rho0LB, 
					 DataType u0LB[LatticeArrangement::getD()],
					 DataType tau 
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
					 , DataType invRho0LB
					 , DataType invTau 
#endif
					 )
{
	rho0LB_ = rho0LB ;
	tau_ = tau ;

	for( unsigned i=0 ; i < LatticeArrangement::getD() ; i++ )
	{
		u0LB_[i] = u0LB[i] ;
	}

#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	invTau_ = invTau ;
	invRho0LB_ = invRho0LB ;
#endif
}



}



#endif

