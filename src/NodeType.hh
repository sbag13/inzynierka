#ifndef NODE_TYPE_HH
#define NODE_TYPE_HH



#include <map>
#include <stdexcept>



namespace microflow
{



inline
HD NodeType::
NodeType()
{
	packedData_ = 0 ;

	boundaryDefinitionIndex_ = 0 ;
	boundaryDefinitionIndex_ = ~boundaryDefinitionIndex_ ;

	placementModifier_ = 0 ;
}



inline
HD NodeType::
NodeType( NodeBaseType nodeBaseType, PlacementModifier placementModifier )
{
	packedData_ = 0 ;

	boundaryDefinitionIndex_ = 0 ;
	placementModifier_ = static_cast<unsigned>(placementModifier) ;

	setBaseType( nodeBaseType ) ;

	if (not microflow::isSolid(nodeBaseType) )
	{
		boundaryDefinitionIndex_ = ~boundaryDefinitionIndex_ ;
	}
}



inline
HD NodeClass NodeType::
getClass() const
{
	return NodeClass::UNKNOWN ;
}



inline
HD NodeBaseType NodeType::
getBaseType() const
{
	return static_cast<NodeBaseType>(baseType_) ;
}



inline
HD PlacementModifier NodeType::
getPlacementModifier() const
{
	return static_cast<PlacementModifier>(placementModifier_) ;
}



inline
HD unsigned char NodeType::
getBoundaryDefinitionIndex() const
{
	return boundaryDefinitionIndex_ ;
}



inline
HD void NodeType::
setBaseType( NodeBaseType nodeBaseType )
{
	baseType_ = static_cast<unsigned short>(nodeBaseType) ;
}



inline
HD void NodeType::
setPlacementModifier( PlacementModifier placementModifier )
{
	placementModifier_ = static_cast<unsigned char>(placementModifier) ;
}



inline
HD void NodeType::
setBoundaryDefinitionIndex( unsigned char boundaryDefinitionIndex )
{
	boundaryDefinitionIndex_ = boundaryDefinitionIndex ;
}



inline
HD bool NodeType::
isSolid() const
{
	return microflow::isSolid( getBaseType() ) ;
}



inline
HD bool NodeType::
isFluid() const
{
	return microflow::isFluid( getBaseType() ) ;
}



inline
HD bool NodeType::
isBoundary() const
{
	return microflow::isBoundary( getBaseType() ) ;
}



inline
HD bool NodeType::
operator == (const NodeBaseType & nodeBaseType) const
{
	return static_cast<unsigned short>(nodeBaseType) == baseType_ ;
}



inline
HD bool NodeType::
operator == (const NodeType & nodeType) const
{
	return (
			( getBaseType()                == nodeType.getBaseType()                ) &&
			( getPlacementModifier()       == nodeType.getPlacementModifier()       ) &&
			( getBoundaryDefinitionIndex() == nodeType.getBoundaryDefinitionIndex() )
		 ) ;
}



inline
HD bool NodeType::
operator != (const NodeBaseType & nodeBaseType) const
{
	return not operator==( nodeBaseType ) ;
}



inline
HD bool NodeType::
operator != (const NodeType & nodeType) const
{
	return not operator==(nodeType) ;
}



inline
std::ostream & 
operator<<(std::ostream& os, const NodeType & nodeType) 
{
	os << "baseType=" ;
	os << (int)nodeType.getBaseType() ;
	os << "(" << toString(nodeType.getBaseType()) << ")" ;
	os << ",placementModifier=" << nodeType.getPlacementModifier() ;
	os << ",boundaryDefinitionIndex=" << (int)nodeType.getBoundaryDefinitionIndex() ;
	return os ;
}



static const std::map< PlacementModifier, std::string > placementModifierNames = 
{
	{PlacementModifier::NONE   , "none"  } ,
	{PlacementModifier::NORTH  , "north" } ,
	{PlacementModifier::SOUTH  , "south" } ,
	{PlacementModifier::EAST   , "east"  } ,
	{PlacementModifier::WEST   , "west"  } ,
	{PlacementModifier::BOTTOM , "bottom"} ,
	{PlacementModifier::TOP    , "top"   } ,
	{PlacementModifier::EXTERNAL_EDGE, "external_edge"} , 
	{PlacementModifier::EXTERNAL_EDGE_PRESSURE_TANGENTIAL, 
		"external_edge_pressure_tangential"} ,
	{PlacementModifier::INTERNAL_EDGE, "internal_edge"} ,
	{PlacementModifier::EXTERNAL_CORNER, "external_corner"} , 
	{PlacementModifier::EXTERNAL_CORNER_PRESSURE_TANGENTIAL, 
		"external_corner_pressure_tangential"} ,
	{PlacementModifier::CORNER_ON_EDGE_AND_PERPENDICULAR_PLANE, 
		"corner_on_edge_and_perpendicular_plane"} ,
	{PlacementModifier::SIZE, "WARNING - size"} 
} ;



inline
const std::string &
toString (PlacementModifier placementModifier)
{
	return placementModifierNames.at(placementModifier) ;
}



inline
std::ostream & 
operator<<(std::ostream & os, const PlacementModifier & placementModifier)
{
	os << (int)placementModifier ;
	os << "(" << toString (placementModifier) << ")" ;
	return os ;
}



template<>
inline
PlacementModifier fromString (const std::string & name)
{
	for (auto it  = placementModifierNames.cbegin() ; 
						it != placementModifierNames.end() ; 
						++it)
	{
		if ( it->second == name ) 
		{
			return it->first ;
		}
	}
	throw std::out_of_range("Name not found in placementModifierNames") ;
}



}
#endif
