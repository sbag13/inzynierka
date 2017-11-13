#ifndef TYPE_NAMES_EXTRACTOR_HH
#define TYPE_NAMES_EXTRACTOR_HH



namespace microflow
{



#define TEMPLATE_TYPE_NAMES_EXTRACTOR                \
template<                                            \
					class LatticeArrangement,                  \
					template<class, class > class FluidModel,  \
					class CollisionModel,                      \
					class DataType                             \
				>



#define TYPE_NAMES_EXTRACTOR \
TypeNamesExtractor< LatticeArrangement, FluidModel, CollisionModel, DataType >



TEMPLATE_TYPE_NAMES_EXTRACTOR
const std::string TYPE_NAMES_EXTRACTOR::
getLatticeArrangementName()
{
	return LatticeArrangement::getName() ;
}



TEMPLATE_TYPE_NAMES_EXTRACTOR
const std::string TYPE_NAMES_EXTRACTOR::
getFluidModelName()
{
	return FluidModel< LatticeArrangement, DataType >::getName() ;
}



TEMPLATE_TYPE_NAMES_EXTRACTOR
const std::string TYPE_NAMES_EXTRACTOR::
getCollisionModelName()
{
	return CollisionModel::getName() ;
}



namespace
{



template< class DataType >
const std::string toString() ;



template<>
const std::string toString<double> ()
{
	return "double" ;
}



template<>
const std::string toString<float> ()
{
	return "single" ;
}



}



TEMPLATE_TYPE_NAMES_EXTRACTOR
const std::string TYPE_NAMES_EXTRACTOR::
getDataTypeName()
{
	return toString<DataType>() ;
}



#undef TYPE_NAMES_EXTRACTOR
#undef TEMPLATE_TYPE_NAMES_EXTRACTOR



}



#endif
