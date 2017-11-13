#ifndef COMPUTATIONAL_ENGINE_HPP
#define COMPUTATIONAL_ENGINE_HPP



#include <string>

#include "Storage.hpp"
#include "TileDataArrangement.hpp"



namespace microflow
{



class ComputationalEngineCPU
{
	public:
		static const std::string getName() { return "CPU" ; }

		//FIXME: Since I have problems with using inherited alias template
		//			 as template template parameter, I am exporting resulting types.
		//template< class T >
		//using StorageType = StorageOnCPU< T > ;
		template
						<
			template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement, 
														class DataType,
														TileDataArrangement DataArrangement
						>
						using LatticeCalculatorType = LatticeCalculator
							<FluidModel, CollisionModel, LatticeArrangement, DataType, StorageOnCPU,
							 DataArrangement> ;
} ;



class ComputationalEngineGPU
{
	public:
		static const std::string getName() { return "GPU" ; }

		//FIXME: Since I have problems with using inherited alias template
		//			 as template template parameter, I am exporting resulting types.
		//template< class T >
		//using StorageType = StorageOnGPU< T > ;
		template
						<
			template<class LatticeArrangement, class DataType> 
														class FluidModel,
														class CollisionModel,
														class LatticeArrangement, 
														class DataType,
														TileDataArrangement DataArrangement
						>
						using LatticeCalculatorType = LatticeCalculator
							< FluidModel, CollisionModel, LatticeArrangement, DataType, StorageOnGPU,
								DataArrangement > ;
} ;



}



#endif
