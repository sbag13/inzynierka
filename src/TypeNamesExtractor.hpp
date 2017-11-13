#ifndef TYPE_NAMES_EXTRACTOR_HPP
#define TYPE_NAMES_EXTRACTOR_HPP



#include <string>



namespace microflow
{



template<
					class LatticeArrangement,
					template<class, class > class FluidModel,
					class CollisionModel,
					class DataType
				>
class TypeNamesExtractor
{
	public:

		static const std::string getLatticeArrangementName() ;
		static const std::string getFluidModelName        () ;
		static const std::string getCollisionModelName    () ;
		static const std::string getDataTypeName          () ;
} ;	



}



#include "TypeNamesExtractor.hh"



#endif
