#include "MRubyInterpreter.hpp"
#include "Exceptions.hpp"

#include <tuple>
#include <type_traits>

#include <mruby/string.h>
#include <mruby/numeric.h>
#include <mruby/array.h>

using namespace std ;

namespace microflow
{
    //NIGDZIE NIE UZYWANE !!!
    // class MRubyException: public std::exception
    // {
    // } mRubyException ;     

    // template<class Type> const std::string typeName() ;
    
    // template <> const std::string typeName<unsigned int>() { return "uint"   ; }
    // template <> const std::string typeName<double>      () { return "double" ; }
    // template <> const std::string typeName<bool>        () { return "bool"   ; }
    // template <> const std::string typeName<std::string> () { return "string" ; }

    template<typename Type>
    Type convertTo (mrb_value rubyVariable) ;

    template<>
    double convertTo<double> (mrb_value rubyVariable)
    {
        if (mrb_float_p (rubyVariable)) {
            return mrb_float (rubyVariable) ;
        }
        else 
        {
            // throw something
        }
    }

    template<>
    string convertTo<string> (mrb_value rubyVariable)
    {
        if (mrb_string_p (rubyVariable)) {
            return RSTRING_PTR (rubyVariable) ;
        }
        else
        {
            // throw something
        }
    }

    template<>
    unsigned convertTo<unsigned> (mrb_value rubyVariable) 
    {
        if (mrb_fixnum_p (rubyVariable)) // można jeszcze sprawdzić czy signed
        {
            return static_cast<unsigned> (mrb_fixnum (rubyVariable)) ;
        }
        else
        {
            // throw something
        }
    }

    template<>
    int convertTo<int> (mrb_value rubyVariable)
    {
        if (mrb_fixnum_p (rubyVariable))
        {
            return  (mrb_fixnum (rubyVariable)) ;
        }
        else
        {
            // throw something
        }
    }

    template<>
    bool convertTo<bool> (mrb_value rubyVariable)
    {
        return mrb_bool (rubyVariable) ;
    }

    MRubyInterpreter* MRubyInterpreter::MRI_instance_ = nullptr ;

    MRubyInterpreter* MRubyInterpreter::
    getMRubyInterpreter ()
    {
        if (!MRI_instance_)
        {
            MRI_instance_ = new MRubyInterpreter () ;
        }
        return MRI_instance_ ;
    }

    MRubyInterpreter::
    MRubyInterpreter ()
    {
        initializeMRubyInterpreter () ;
    }

    MRubyInterpreter::
    ~MRubyInterpreter ()
    {
        if (state_ != nullptr) {
            mrbc_context_free (state_, context_) ;
            closeMRubyInterpreter () ;
        } 
        state_ = nullptr ;
    }

    void MRubyInterpreter::
    runScript (const string& code)
    {
        parser_ = mrb_parse_string (state_, code.c_str(), context_) ;
        proc_ = mrb_generate_code (state_, parser_) ;
        mrb_pool_close (parser_->pool) ;
        value_ = mrb_run (state_, proc_, mrb_top_self(state_)) ;

        if (state_->exc) {
            logger << "ERROR in Ruby\n" ;
            mrb_value lasterr = mrb_obj_value (state_->exc) ;

            //class
            mrb_value klass = mrb_class_path (state_, mrb_obj_class(state_, lasterr)) ;
            logger << "class = " << convertTo<string> (klass) << endl ;

            //message
            logger << "message = " << convertTo<string> (lasterr) << endl ;

            THROW ("Ruby") ;
        }
    }

    template<class VariableType >
    VariableType MRubyInterpreter::
    getMRubyVariable( const std::string & variableName )
    {
        mrb_value mrb_string = mrb_str_new_cstr (state_, variableName.c_str()) ;
        mrb_sym symbol = mrb_intern_str (state_, mrb_string) ;
        mrb_value rubyVariable = mrb_gv_get (state_, symbol) ;
        VariableType variable = convertTo<VariableType>( rubyVariable ) ;
        if (mrb_equal (state_, rubyVariable, mrb_nil_value ()))     //check if the variable exists
        {
            cout << "nie ma takiej zmiennej" << endl;   //TO REMOVE
            //jakiś throw
        }
        return variable ;
    }

    // Used in methods called by Ruby interpreter - these methods must be static.
    static NodeLayout * nodeLayoutPtr = NULL ;
    static ModificationRhoU * modificationsRhoUPtr = NULL ;

    static mrb_value setNodeBaseType (mrb_state * state, mrb_value self) 
    {
        //if (mrb_get_argc == 4) //TO_DO

        mrb_int mrb_nodeZ ;
        mrb_int mrb_nodeX ;
        mrb_int mrb_nodeY ;
        mrb_value mrb_nodeBaseTypeName ;

        mrb_get_args (state, "iiiz", &mrb_nodeX, &mrb_nodeY, &mrb_nodeZ, &mrb_nodeBaseTypeName) ;

        unsigned nodeX = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeX)) ;
        unsigned nodeY = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeY)) ;
        unsigned nodeZ = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeZ)) ;
        string nodeBaseTypeName = convertTo<string> (mrb_nodeBaseTypeName) ;

        auto node = nodeLayoutPtr->getNodeType (nodeX,nodeY,nodeZ) ;
	    node.setBaseType (fromString<NodeBaseType> (nodeBaseTypeName)) ;
	    nodeLayoutPtr->setNodeType (nodeX, nodeY, nodeZ, node) ;
        return mrb_nil_value () ;
    }

    static mrb_value setNodePlacementModifier (mrb_state * state, mrb_value self)
    {
        //if (mrb_get_argc == 4) //TO_DO

        mrb_int mrb_nodeX ;           //to wczytywanie argumentów mozna wywalic do funkcji
        mrb_int mrb_nodeY ;
        mrb_int mrb_nodeZ ;
        mrb_value mrb_placementModifierName ;

        mrb_get_args (state, "iiiS", &mrb_nodeX, &mrb_nodeY, &mrb_nodeZ, &mrb_placementModifierName) ;

        unsigned nodeX = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeX)) ;
        unsigned nodeY = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeY)) ;
        unsigned nodeZ = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeZ)) ;
        string placementModifierName = convertTo<string> (mrb_placementModifierName) ;

        auto node = nodeLayoutPtr->getNodeType (nodeX,nodeY,nodeZ) ;
	    node.setPlacementModifier (fromString<PlacementModifier> (placementModifierName)) ;
	    nodeLayoutPtr->setNodeType (nodeX,nodeY,nodeZ, node) ;

	    return mrb_nil_value () ;
    }

    static mrb_value setNodeRhoPhysical (mrb_state * state, mrb_value self) 
    {
        mrb_int mrb_nodeX ;           //to wczytywanie argumentów mozna wywalic do funkcji
        mrb_int mrb_nodeY ;
        mrb_int mrb_nodeZ ;
        mrb_float mrb_rhoPhysical;

        mrb_get_args (state, "iiif", &mrb_nodeX, &mrb_nodeY, &mrb_nodeZ, &mrb_rhoPhysical) ;

        unsigned nodeX = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeX)) ;
        unsigned nodeY = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeY)) ;
        unsigned nodeZ = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeZ)) ;
        double rhoPhysical = convertTo<double> (mrb_float_value (state, mrb_rhoPhysical)) ;
        
        modificationsRhoUPtr->addRhoPhysical (Coordinates (nodeX, nodeY, nodeZ), rhoPhysical) ;

        return mrb_nil_value () ;
    }

    static mrb_value setNodeRhoBoundaryPhysical (mrb_state * state, mrb_value self) 
    {
        mrb_int mrb_nodeX ;           //to wczytywanie argumentów mozna wywalic do funkcji
        mrb_int mrb_nodeY ;
        mrb_int mrb_nodeZ ;
        mrb_float mrb_rhoPhysical;

        mrb_get_args (state, "iiif", &mrb_nodeX, &mrb_nodeY, &mrb_nodeZ, &mrb_rhoPhysical) ;


        unsigned nodeX = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeX)) ;
        unsigned nodeY = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeY)) ;
        unsigned nodeZ = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeZ)) ;
        double rhoPhysical = convertTo<double> (mrb_float_value (state, mrb_rhoPhysical)) ;

        modificationsRhoUPtr->addRhoBoundaryPhysical (Coordinates (nodeX, nodeY, nodeZ), rhoPhysical) ;

        return mrb_nil_value () ;
    }

    static mrb_value setNodeUPhysical (mrb_state * state, mrb_value self) 
    {
        mrb_int mrb_nodeX ;           //to wczytywanie argumentów mozna wywalic do funkcji
        mrb_int mrb_nodeY ;
        mrb_int mrb_nodeZ ;
        mrb_value mrb_uPhysical;

        mrb_get_args (state, "iiif", &mrb_nodeX, &mrb_nodeY, &mrb_nodeZ, &mrb_uPhysical) ;

        unsigned nodeX = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeX)) ;
        unsigned nodeY = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeY)) ;
        unsigned nodeZ = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeZ)) ;
        double ux = convertTo<double> (mrb_ary_entry (mrb_uPhysical, 0)) ;
	    double uy = convertTo<double> (mrb_ary_entry (mrb_uPhysical, 1)) ;
	    double uz = convertTo<double> (mrb_ary_entry (mrb_uPhysical, 2)) ;

        modificationsRhoUPtr->addUPhysical (Coordinates (nodeX, nodeY, nodeZ), ux,uy,uz) ;

        return mrb_nil_value () ;
    }

    static mrb_value setNodeUBoundaryPhysical (mrb_state * state, mrb_value self) 
    {
        mrb_int mrb_nodeX ;           //to wczytywanie argumentów mozna wywalic do funkcji
        mrb_int mrb_nodeY ;
        mrb_int mrb_nodeZ ;
        mrb_value mrb_uPhysical;

        mrb_get_args (state, "iiif", &mrb_nodeX, &mrb_nodeY, &mrb_nodeZ, &mrb_uPhysical) ;

        unsigned nodeX = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeX)) ;
        unsigned nodeY = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeY)) ;
        unsigned nodeZ = convertTo<unsigned> (mrb_fixnum_value (mrb_nodeZ)) ;
        double ux = convertTo<double> (mrb_ary_entry (mrb_uPhysical, 0)) ;
	    double uy = convertTo<double> (mrb_ary_entry (mrb_uPhysical, 1)) ;
	    double uz = convertTo<double> (mrb_ary_entry (mrb_uPhysical, 2)) ;

        modificationsRhoUPtr->addUBoundaryPhysical (Coordinates (nodeX, nodeY, nodeZ), ux,uy,uz) ;

        return mrb_nil_value () ;
    }

    mrb_value createMRubyObject (mrb_state* mrb, const std::string& className)
    {
        struct RClass *mrb_class ;
        mrb_value mrb_object ;
        mrb_class = mrb_define_class(mrb, className.c_str (), mrb->object_class) ;
        mrb_object = mrb_obj_new (mrb, mrb_class, 0, NULL) ;

        return mrb_object ;
    }

    mrb_sym mrb_symbol_from (const std::string& symbol)     //jesli jest singleton
    {
        mrb_state* state = MRubyInterpreter::getMRubyInterpreter () -> getState () ;
        return mrb_intern_str (state, mrb_str_new_cstr (state, "@width")) ;
    }

    //TEST mozna zrobic evaluation ...

    static mrb_value getNode (mrb_state * state, mrb_value self) 
    {
        mrb_int mrb_X ;           //to wczytywanie argumentów mozna wywalic do funkcji
        mrb_int mrb_Y ;
        mrb_int mrb_Z ;

        mrb_get_args (state, "iiif", &mrb_X, &mrb_Y, &mrb_Z) ;

        unsigned x = convertTo<unsigned> (mrb_fixnum_value (mrb_X)) ;
        unsigned y = convertTo<unsigned> (mrb_fixnum_value (mrb_Y)) ;
        unsigned z = convertTo<unsigned> (mrb_fixnum_value (mrb_Z)) ;

        mrb_value node = createMRubyObject (state, "Node") ;

        NodeType nodeType ;
        Coordinates coordinates (x, y, z) ;

        Size size = nodeLayoutPtr->getSize () ;
        if (size.areCoordinatesInLimits (coordinates))
        {
            nodeType = nodeLayoutPtr->getNodeType (coordinates) ;
        }
        else
        {
            logger << "WARNING: Can not get node type at " << coordinates 
					 << ", coordinates outside of " << size << "\n" ;
		    return mrb_nil_value () ;
        }

        std::string baseTypeName = toString (nodeType.getBaseType()) ;
        std::string placementModifierName = toString (nodeType.getPlacementModifier()) ;

        // mrb_sym mrb_symbol_baseType = mrb_intern_str (state, mrb_str_new_cstr (state, "@baseType")) ;
        // mrb_sym mrb_symbol_placementModifier = mrb_intern_str (state, mrb_str_new_cstr (state, "@placementModifier")) ;

        // mrb_iv_set(state, node, mrb_symbol_baseType, mrb_str_new_cstr (state, baseTypeName.c_str())) ;
        // mrb_iv_set(state, node, mrb_symbol_placementModifier, mrb_str_new_cstr (state, placementModifierName.c_str())) ;

        mrb_iv_set(state, node, mrb_symbol_from ("@baseType"), mrb_str_new_cstr (state, baseTypeName.c_str())) ;        //mrb_str_new_cstr jak singleton -> funkcja
        mrb_iv_set(state, node, mrb_symbol_from ("@placementModifier"), mrb_str_new_cstr (state, placementModifierName.c_str())) ;

        return node ;
    }

    static mrb_value getSize (mrb_state * state, mrb_value self) 
    {
        mrb_value size = createMRubyObject (state, "Size") ;

        Size nodeLayoutSize = nodeLayoutPtr->getSize () ;

        // mrb_sym mrb_symbol_width = mrb_intern_str (state, mrb_str_new_cstr (state, "@width")) ;
        // mrb_sym mrb_symbol_height = mrb_intern_str (state, mrb_str_new_cstr (state, "@height")) ;
        // mrb_sym mrb_symbol_depth = mrb_intern_str (state, mrb_str_new_cstr (state, "@depth")) ;

        // mrb_iv_set (state, size, mrb_symbol_width, mrb_fixnum_value (nodeLayoutSize.getWidth ())) ;
        // mrb_iv_set (state, size, mrb_symbol_height, mrb_fixnum_value (nodeLayoutSize.getHeight ())) ;
        // mrb_iv_set (state, size, mrb_symbol_depth, mrb_fixnum_value (nodeLayoutSize.getDepth ())) ;

        mrb_iv_set (state, size, mrb_symbol_from ("@width"), mrb_fixnum_value (nodeLayoutSize.getWidth ())) ;
        mrb_iv_set (state, size, mrb_symbol_from ("@height"), mrb_fixnum_value (nodeLayoutSize.getHeight ())) ;
        mrb_iv_set (state, size, mrb_symbol_from ("@depth"), mrb_fixnum_value (nodeLayoutSize.getDepth ())) ;

        return size ;
    }

    static void
    initializeRubyModifyLayout()
    {
        mrb_state* state = MRubyInterpreter::getMRubyInterpreter () -> getState () ;

        mrb_define_method (state, state->kernel_module, "setNodeBaseType", setNodeBaseType, MRB_ARGS_REQ(4)) ;
        mrb_define_method (state, state->kernel_module, "setNodePlacementModifier", setNodePlacementModifier, MRB_ARGS_REQ(4)) ;
        mrb_define_method (state, state->kernel_module, "setNodeRhoPhysical", setNodeRhoPhysical, MRB_ARGS_REQ(4)) ;
        mrb_define_method (state, state->kernel_module, "setNodeRhoBoundaryPhysical", setNodeRhoBoundaryPhysical, MRB_ARGS_REQ(4)) ;
        mrb_define_method (state, state->kernel_module, "setNodeUPhysical", setNodeUPhysical, MRB_ARGS_REQ(4)) ;
        mrb_define_method (state, state->kernel_module, "setNodeUBoundaryPhysical", setNodeUBoundaryPhysical, MRB_ARGS_REQ(4)) ;
        mrb_define_method (state, state->kernel_module, "getNode", getNode, MRB_ARGS_REQ(3)) ;
        mrb_define_method (state, state->kernel_module, "getSize", getSize, MRB_ARGS_NONE()) ;


        //mrb_define_method (state, state->kernel_module, "testFN", testFN, MRB_ARGS_REQ(1)) ;
    }

    ModificationRhoU MRubyInterpreter::
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

    void MRubyInterpreter::
    initializeMRubyInterpreter()
    {
        state_ = mrb_open() ;
        context_ = mrbc_context_new(state_) ;
        if (!state_) 
        {
            //handle error
        }//vs code does not see header file in the same
    }

    void MRubyInterpreter::
    closeMRubyInterpreter()
    {
        mrb_close(state_) ;
    }

    const mrb_value MRubyInterpreter::  //tylko do testów, nie musi być publiczne
    getValue () const
    {
        return this->value_ ;
    }

    mrb_state* MRubyInterpreter::
    getState ()
    {
        return this->state_ ;
    }

    void test ()
    {
        ifstream str("script.rb");
        string code((istreambuf_iterator<char>(str)), istreambuf_iterator<char>());

        MRubyInterpreter* ptr = MRubyInterpreter::getMRubyInterpreter() ;
        
        initializeRubyModifyLayout () ;
        ptr->runScript (code) ;
    }
}

