#ifndef NODE_TYPE_HPP
#define NODE_TYPE_HPP



#include <ostream>

#include "gpuTools.hpp"
#include "microflowTools.hpp"
#include "NodeBaseType.hpp"



namespace microflow
{


// TODO: http://stackoverflow.com/questions/9510514/integer-range-based-template-specialisation

enum class NodeClass 
{ 
	SOLID    = 0, 
	FLUID    = 1, 
	BOUNDARY = 2, 
	UNKNOWN  = 3 
} ;



enum class PlacementModifier
{
	NONE   = 0,

	NORTH  = 1,
	SOUTH  = 2,
	EAST   = 3,
	WEST   = 4,
	BOTTOM = 5,
	TOP    = 6,

	// modifiers for velocity nodes only
	EXTERNAL_EDGE                              = 7,
	EXTERNAL_EDGE_PRESSURE_TANGENTIAL          = 8,
	INTERNAL_EDGE                              = 9,
	EXTERNAL_CORNER                            = 10,
	EXTERNAL_CORNER_PRESSURE_TANGENTIAL        = 11,
	CORNER_ON_EDGE_AND_PERPENDICULAR_PLANE     = 12,

	SIZE
} ;



template<>
PlacementModifier fromString (const std::string & name) ;



class NodeType
{
	public:
		
		typedef unsigned short PackedDataType ;

		HD NodeType() ;
		HD NodeType( NodeBaseType nodeBaseType, 
								 PlacementModifier placementModifier = PlacementModifier::NONE ) ;

		HD NodeBaseType  getBaseType() const ;
		HD PlacementModifier getPlacementModifier() const ;
		HD unsigned char getBoundaryDefinitionIndex() const ;

		HD void setBaseType( NodeBaseType nodeBaseType ) ;
		HD void setPlacementModifier( PlacementModifier placementModifier ) ;
		HD void setBoundaryDefinitionIndex( unsigned char boundaryDefinitionIndex ) ;

		//TODO: Remove ? Unused method.
		HD NodeClass getClass() const ;

		HD bool isSolid() const ;
		HD bool isFluid() const ;
		HD bool isBoundary() const ;

		HD bool operator == (const NodeBaseType & nodeBaseType) const ;
		HD bool operator == (const NodeType & nodeType) const ;

		HD bool operator != (const NodeBaseType & nodeBaseType) const ;
		HD bool operator != (const NodeType & nodeType) const ;


		// Needed for __ldg(), currently in work.
		HD PackedDataType & packedData() { return packedData_ ; }

	private:

		static constexpr unsigned BITS_PER_BASE_TYPE = 
								sizeInBits(static_cast<unsigned>(NodeBaseType::SIZE)) ;

		static constexpr unsigned BITS_PER_PLACEMENT_MODIFIER = 
								sizeInBits(static_cast<unsigned>(PlacementModifier::SIZE)) ;
		static constexpr unsigned BITS_PER_BOUNDARY_DEFINITION_INDEX = 6 ;

		static constexpr unsigned BITS_PER_NODE_TYPE = BITS_PER_BASE_TYPE + 
			                                         BITS_PER_PLACEMENT_MODIFIER + 
			                                         BITS_PER_BOUNDARY_DEFINITION_INDEX ;
		
		// Needed for __ldg(), currently in work.
		union
		{
			struct 
			{
				PackedDataType baseType_                : BITS_PER_BASE_TYPE                 ; 
				PackedDataType placementModifier_       : BITS_PER_PLACEMENT_MODIFIER        ;
				PackedDataType boundaryDefinitionIndex_ : BITS_PER_BOUNDARY_DEFINITION_INDEX ;
			} ;
			PackedDataType packedData_ ;
		} ;
} ;



const std::string & toString( PlacementModifier placementModifier ) ;

std::ostream & operator<<(std::ostream & os, const NodeType & nodeType) ;
std::ostream & operator<<(std::ostream & os, const PlacementModifier & placementModifier) ;



}



#include "NodeType.hh"



#endif
