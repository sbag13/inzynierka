#include "MRubyInterpreter.hpp"
#include "Exceptions.hpp"

#include <mruby/string.h>
#include <mruby/numeric.h>

using namespace std ;

int fn1 (){return 1;};

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

    static mrb_value setNodeBaseType (mrb_state * state, mrb_value self)  //mrb_value x, mrb_value y, mrb_value z, mrb_value baseType
    {
        //if (mrb_get_argc == 4) //TO_DO

        mrb_value mrb_nodeX ;
        mrb_value mrb_nodeY ;
        mrb_value mrb_nodeZ ;
        mrb_value mrb_nodeBaseTypeName ;

        mrb_get_args (state, "iiiz", &mrb_nodeX, &mrb_nodeY, &mrb_nodeZ, &mrb_nodeBaseTypeName) ;

        unsigned nodeX = convertTo<unsigned> (mrb_nodeX) ;
        unsigned nodeY = convertTo<unsigned> (mrb_nodeY) ;
        unsigned nodeZ = convertTo<unsigned> (mrb_nodeZ) ;
        string nodeBaseTypeName = convertTo<string> (mrb_nodeBaseTypeName) ;
    }

    static void
    initializeRubyModifyLayout()
    {

    }

    ModificationRhoU MRubyInterpreter::
    modifyNodeLayout (NodeLayout & nodeLayout, const std::string & rubyCode)
    {
        initializeRubyModifyLayout() ;
    }

    void MRubyInterpreter::
    initializeMRubyInterpreter()
    {
        state_ = mrb_open() ;
        context_ = mrbc_context_new(state_) ;
        if (!state_) 
        {
            //handle error
        }
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

    void test ()
    {
        ifstream str("script.rb");
        string code((istreambuf_iterator<char>(str)), istreambuf_iterator<char>());

        MRubyInterpreter* ptr = MRubyInterpreter::getMRubyInterpreter() ;
        ptr->runScript (code) ;
        string a = ptr->getMRubyVariable<string> ("$var") ;
        cout << a ;
    }
}

