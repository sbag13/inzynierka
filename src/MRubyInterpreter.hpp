#ifndef MRUBY_INTERPRETER_HPP
#define MRUBY_INTERPRETER_HPP

#include <memory>
#include <string>
#include <mruby.h>
#include <mruby/compile.h>
#include <mruby/proc.h>
#include <mruby/variable.h>

#include "Coordinates.hpp"
#include "NodeLayout.hpp"
#include "ModificationRhoU.hpp"

#include "NodeBaseType.hpp" 

namespace microflow
{
    class MRubyInterpreter
    {       
    public:

        static MRubyInterpreter* getMRubyInterpreter () ;
        ~MRubyInterpreter () ;
        void runScript (const std::string&) ;

        template<class VariableType >
        VariableType getMRubyVariable (const std::string & variableName) ;
        
        ModificationRhoU modifyNodeLayout (NodeLayout & nodeLayout, const std::string & rubyCode) ;

        const mrb_value getValue () const ; // tylko do testów, nie musi być publiczne
        mrb_state* getState () ;

    private:
        MRubyInterpreter () ; 
        void initializeMRubyInterpreter () ;
        void closeMRubyInterpreter ();

        mrb_state* state_ = nullptr ;
        mrbc_context * context_ ;
        mrb_value value_ ;
        struct mrb_parser_state * parser_ ;
        struct RProc * proc_ ;
        static MRubyInterpreter* MRI_instance_ ; 
    } ;

    void test () ;
}

#endif