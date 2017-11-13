#ifndef RUBY_INTERPRETER_HPP
#define RUBY_INTERPRETER_HPP



#include <string>
#include <iostream>
#include <ext/stdio_filebuf.h>
#include <array>

#include "Coordinates.hpp"
#include "NodeLayout.hpp"
#include "ModificationRhoU.hpp"



//TODO: consider using mruby -  "it allows you to run multiple copies of 
//			Ruby inside one program". Look at https://github.com/mruby/mruby


namespace microflow
{


/*
	Singleton class, because Ruby interpreter can not be unloaded.
*/
class RubyInterpreter
{
	public:
		
		static RubyInterpreter * getInterpreter() ;

		~RubyInterpreter() ;

		void runScript( const char * script ) ;

		template<class VariableType >
		VariableType getRubyVariable( const std::string & variableName ) ;

		ModificationRhoU modifyNodeLayout (NodeLayout & nodeLayout, const std::string & rubyCode) ;


	private:

		RubyInterpreter() ;
		void initializeRubyInterpreter() ;

		static RubyInterpreter * instance_ ;

} ;



}



#endif
