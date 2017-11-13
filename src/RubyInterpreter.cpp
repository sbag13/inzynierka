#include <ruby.h>
#include <exception>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <algorithm>



#include "RubyInterpreter.hpp"
#include "Exceptions.hpp"



using namespace std ;



namespace microflow
{



class RubyException: public std::exception
{
} rubyException ;



template<class Type> const std::string typeName() ;

template <> const std::string typeName<unsigned int>() { return "uint"   ; }
template <> const std::string typeName<double>      () { return "double" ; }
template <> const std::string typeName<bool>        () { return "bool"   ; }
template <> const std::string typeName<std::string> () { return "string" ; }



template<class Type >
Type convertTo(VALUE rubyVariable) ;



template<> double convertTo<double>(VALUE rubyVariable)
{
	return NUM2DBL(rubyVariable) ;
}



template<> unsigned int convertTo<unsigned int>(VALUE rubyVariable)
{
	return NUM2UINT(rubyVariable) ;
}



template<> bool convertTo<bool>(VALUE rubyVariable)
{
	return RTEST(rubyVariable) ;
}



template<> std::string convertTo<std::string>(VALUE rubyVariable)
{
	return StringValueCStr(rubyVariable) ;
}




RubyInterpreter* RubyInterpreter::instance_ = NULL ;


RubyInterpreter * RubyInterpreter::
getInterpreter() 
{
  if (!instance_) 
	{
    instance_ = new RubyInterpreter() ;
  }
  return instance_ ;
}


RubyInterpreter::
RubyInterpreter()
{
	initializeRubyInterpreter() ;
}



RubyInterpreter::
~RubyInterpreter()
{
}



void RubyInterpreter::
runScript(const char * script)
{
	int err ;

	rb_protect((VALUE (*)(VALUE))rb_eval_string, (VALUE)script, &err);

	// Error message according to http://aeditor.rubyforge.org/ruby_cplusplus/
	if (0 != err)
	{
		logger << "ERROR in Ruby\n" ;

		VALUE lasterr = rb_gv_get("$!");

		// class
		VALUE klass = rb_class_path(CLASS_OF(lasterr));
		logger << "class = " << RSTRING_PTR(klass) << endl; 

		// message
		VALUE message = rb_obj_as_string(lasterr);
		logger << "message = " << RSTRING_PTR(message) << endl;

		THROW ("Ruby") ;
	} 
}



template<class VariableType >
VariableType RubyInterpreter::
getRubyVariable( const std::string & variableName )
{
	VALUE rubyVariable = rb_gv_get(variableName.c_str()) ;
	VariableType variable = convertTo<VariableType>( rubyVariable ) ;
	return variable ;
}



template double       RubyInterpreter::getRubyVariable<double>
								     									(const std::string & variableName ) ;
template bool         RubyInterpreter::getRubyVariable<bool>
										   								(const std::string & variableName ) ;
template std::string  RubyInterpreter::getRubyVariable<std::string>
																			(const std::string & variableName ) ;
template unsigned int RubyInterpreter::getRubyVariable<unsigned int>
																			(const std::string & variableName ) ;



// Used in methods called by Ruby interpreter - these methods must be static.
static NodeLayout * nodeLayoutPtr = NULL ;
static ModificationRhoU * modificationsRhoUPtr = NULL ;



/*
			Methods called by Ruby interpreter.
*/
static VALUE setNodeBaseType (VALUE self, VALUE x, VALUE y, VALUE z, VALUE baseType)
{
	if (self) {} ; // Avoid compiler warning

	unsigned nodeX = convertTo<unsigned> (x) ;
	unsigned nodeY = convertTo<unsigned> (y) ;
	unsigned nodeZ = convertTo<unsigned> (z) ;
	std::string nodeBaseTypeName = convertTo<std::string> (baseType) ;

	auto node = nodeLayoutPtr->getNodeType (nodeX,nodeY,nodeZ) ;
	node.setBaseType (fromString<NodeBaseType> (nodeBaseTypeName)) ;
	nodeLayoutPtr->setNodeType (nodeX,nodeY,nodeZ, node) ;

	return Qnil ;
}



static VALUE setNodePlacementModifier (VALUE self, VALUE x, VALUE y, VALUE z, 
																			 VALUE placementModifier)
{
	if (self) {} ; // Avoid compiler warning

	unsigned nodeX = convertTo<unsigned> (x) ;
	unsigned nodeY = convertTo<unsigned> (y) ;
	unsigned nodeZ = convertTo<unsigned> (z) ;
	std::string placementModifierName = convertTo<std::string> (placementModifier) ;

	auto node = nodeLayoutPtr->getNodeType (nodeX,nodeY,nodeZ) ;
	node.setPlacementModifier (fromString<PlacementModifier> (placementModifierName)) ;
	nodeLayoutPtr->setNodeType (nodeX,nodeY,nodeZ, node) ;

	return Qnil ;
}



static VALUE setNodeRhoPhysical (VALUE self, VALUE x, VALUE y, VALUE z, 
																 VALUE rhoPhysical)
{
	if (self) {} ; // Avoid compiler warning

	unsigned nodeX = convertTo<unsigned> (x) ;
	unsigned nodeY = convertTo<unsigned> (y) ;
	unsigned nodeZ = convertTo<unsigned> (z) ;
	double rho = convertTo<double> (rhoPhysical) ;

	modificationsRhoUPtr->addRhoPhysical (Coordinates (nodeX, nodeY, nodeZ), rho) ;

	return Qnil ;
}



static VALUE setNodeRhoBoundaryPhysical (VALUE self, VALUE x, VALUE y, VALUE z, 
																 				 VALUE rhoPhysical)
{
	if (self) {} ; // Avoid compiler warning

	unsigned nodeX = convertTo<unsigned> (x) ;
	unsigned nodeY = convertTo<unsigned> (y) ;
	unsigned nodeZ = convertTo<unsigned> (z) ;
	double rho = convertTo<double> (rhoPhysical) ;

	modificationsRhoUPtr->addRhoBoundaryPhysical (Coordinates (nodeX, nodeY, nodeZ), rho) ;

	return Qnil ;
}



static VALUE setNodeUPhysical (VALUE self, VALUE x, VALUE y, VALUE z, 
															 VALUE uPhysical)
{
	if (self) {} ; // Avoid compiler warning

	unsigned nodeX = convertTo<unsigned> (x) ;
	unsigned nodeY = convertTo<unsigned> (y) ;
	unsigned nodeZ = convertTo<unsigned> (z) ;
	double ux = convertTo<double> ( rb_ary_entry(uPhysical, 0) ) ;
	double uy = convertTo<double> ( rb_ary_entry(uPhysical, 1) ) ;
	double uz = convertTo<double> ( rb_ary_entry(uPhysical, 2) ) ;

	modificationsRhoUPtr->addUPhysical (Coordinates (nodeX, nodeY, nodeZ), ux,uy,uz) ;

	return Qnil ;
}



static VALUE setNodeUBoundaryPhysical (VALUE self, VALUE x, VALUE y, VALUE z, 
															 				 VALUE uPhysical)
{
	if (self) {} ; // Avoid compiler warning

	unsigned nodeX = convertTo<unsigned> (x) ;
	unsigned nodeY = convertTo<unsigned> (y) ;
	unsigned nodeZ = convertTo<unsigned> (z) ;
	double ux = convertTo<double> ( rb_ary_entry(uPhysical, 0) ) ;
	double uy = convertTo<double> ( rb_ary_entry(uPhysical, 1) ) ;
	double uz = convertTo<double> ( rb_ary_entry(uPhysical, 2) ) ;

	modificationsRhoUPtr->addUBoundaryPhysical (Coordinates (nodeX, nodeY, nodeZ), ux,uy,uz) ;

	return Qnil ;
}



VALUE createRubyObject (std::string className)
{
	ID symClas = rb_intern(className.c_str());
	VALUE clas = rb_const_get(rb_cObject, symClas);
	VALUE argv[0];
	
	return rb_class_new_instance(0, argv, clas);
}



static VALUE getNode (VALUE self, VALUE x, VALUE y, VALUE z)
{
	if (self) {} ; // Avoid compiler warning

	VALUE node = createRubyObject ("Node") ;

	NodeType nodeType ;
	Coordinates coordinates (convertTo<unsigned>(x), 
													 convertTo<unsigned>(y), 
													 convertTo<unsigned>(z) ) ;
	Size size = nodeLayoutPtr->getSize() ;
	if (size.areCoordinatesInLimits (coordinates))
	{
		nodeType = nodeLayoutPtr->getNodeType (coordinates) ;
	}
	else
	{
		logger << "WARNING: Can not get node type at " << coordinates 
					 << ", coordinates outside of " << size << "\n" ;
		return Qnil ;
	}

	std::string baseTypeName = toString (nodeType.getBaseType()) ;
	std::string placementModifierName = toString (nodeType.getPlacementModifier()) ;

	rb_iv_set(node, "@baseType", rb_str_new2 (baseTypeName.c_str())) ;
	rb_iv_set(node, "@placementModifier", rb_str_new2 (placementModifierName.c_str())) ;

	return node ;
}



static VALUE getSize (VALUE self)
{
	if (self) {} ; // Avoid compiler warning

	VALUE size = createRubyObject ("Size") ;

	Size nodeLayoutSize = nodeLayoutPtr->getSize() ;

	rb_iv_set(size, "@width" , INT2FIX (nodeLayoutSize.getWidth() ) ) ;
	rb_iv_set(size, "@height", INT2FIX (nodeLayoutSize.getHeight()) ) ;
	rb_iv_set(size, "@depth" , INT2FIX (nodeLayoutSize.getDepth() ) ) ;

	return size ;
}



static void
initializeRubyModifyLayout()
{
	rb_define_global_function ("setNodeBaseType", 
						      					 reinterpret_cast<VALUE(*)(...)> (setNodeBaseType), 4) ;
	rb_define_global_function ("setNodePlacementModifier", 
						      					 reinterpret_cast<VALUE(*)(...)> (setNodePlacementModifier), 4) ;
	rb_define_global_function ("setNodeRhoPhysical", 
						      					 reinterpret_cast<VALUE(*)(...)> (setNodeRhoPhysical), 4) ;
	rb_define_global_function ("setNodeRhoBoundaryPhysical", 
						      					 reinterpret_cast<VALUE(*)(...)> (setNodeRhoBoundaryPhysical), 4) ;
	rb_define_global_function ("setNodeUPhysical", 
						      					 reinterpret_cast<VALUE(*)(...)> (setNodeUPhysical), 4) ;
	rb_define_global_function ("setNodeUBoundaryPhysical", 
						      					 reinterpret_cast<VALUE(*)(...)> (setNodeUBoundaryPhysical), 4) ;
	rb_define_global_function ("getNode", 
														 reinterpret_cast<VALUE(*)(...)> (getNode), 3) ;
	rb_define_global_function ("getSize", 
														 reinterpret_cast<VALUE(*)(...)> (getSize), 0) ;
}



ModificationRhoU RubyInterpreter::
modifyNodeLayout (NodeLayout & nodeLayout, const std::string & rubyCode)
{
	initializeRubyModifyLayout() ;

	nodeLayoutPtr = &nodeLayout ;
	ModificationRhoU modifications ;
	modificationsRhoUPtr = &modifications ;

	//TODO: Use xxd.
	#define STRINGIFY(x) #x
	const char * script = //TODO: should I move it to RubyScripts.hpp ?
			#include "modifyNodeLayout.rb"
			;
	#undef STRINGIFY

	std::string code = script + rubyCode ;

	runScript (code.c_str()) ;

	nodeLayoutPtr = NULL ;
	modificationsRhoUPtr = NULL ;

	return modifications ;
}



void RubyInterpreter::
initializeRubyInterpreter()
{
	ruby_init();
	ruby_init_loadpath();
	ruby_script("configuration_reader"); 	/* sets name in error messages */
}



}
