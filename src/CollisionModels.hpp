#ifndef COLLISION_MODELS_HPP
#define COLLISION_MODELS_HPP



#include <string>



namespace microflow
{



class CollisionModelBase
{
	public:
		static constexpr bool isBGK = false ;
		static constexpr bool isMRT = false ;
} ;



class CollisionModelBGK
: public CollisionModelBase
{
	public:

		static constexpr bool isBGK = true ;

		static const std::string getName() { return "CollisionModelBGK" ; } ;

} ;



class CollisionModelMRT
: public CollisionModelBase
{
	public:

		static constexpr bool isMRT = true ;

		static const std::string getName() { return "CollisionModelMRT" ; } ;


} ;



}



#endif
