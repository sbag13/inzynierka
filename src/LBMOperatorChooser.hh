#ifndef LBM_OPERATOR_CHOOSER_HH
#define LBM_OPERATOR_CHOOSER_HH



namespace microflow
{



HD_WARNING_DISABLE
template< class Calculator, class Element >
HD
void InitializatorAtEquilibrium::
apply( Calculator & calculator, Element & element )
{
	calculator.initializeAtEquilibrium( element ) ;
}



HD_WARNING_DISABLE
template< class Calculator, class Element >
HD
void Collider::
apply( Calculator & calculator, Element & element )
{
	calculator.collide( element ) ;
}



HD_WARNING_DISABLE
template< class Calculator, class Element >
HD
void Propagator::
apply( Calculator & calculator, Element & element )
{
	calculator.propagate( element ) ;
}



// Only for CPU version, GPU version needs information about grid configuration.
template< class Calculator, class Element >
void BoundaryProcessor::
apply( Calculator & calculator, Element & element )
{
#ifndef __CUDA_ARCH__
	calculator.processBoundary( element ) ;
#endif
}



template< class Calculator, class Element >
void RhoBB2Calculator::
apply (Calculator & calculator, Element & element)
{
#ifndef __CUDA_ARCH__
	calculator.computeRhoForBB2Nodes (element) ;
#endif
}



}
#endif
