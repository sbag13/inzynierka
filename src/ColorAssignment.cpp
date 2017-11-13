#include <iostream>
#include <string>

#include "Logger.hpp"
#include "microflowTools.hpp"
#include "ColorAssignment.hpp"
#include "Exceptions.hpp"



using namespace std ;



namespace microflow
{



bool
ColorAssignment::
readElement (const string & elementName, istream & stream)
{
	bool result = true ;

	if (not BoundaryDescription::readElement (elementName, stream))
	{
		if ("color" == elementName)
		{
			readChar(stream, '=') ;
			readChar(stream, '(') ;
			stream >> std::dec ;
			unsigned c ;
			stream >> c ;
			color_.red = c  ;
			readChar(stream, ',') ;
			stream >> c ;
			color_.green = c ;
			readChar(stream, ',') ;
			stream >> c ;
			color_.blue = c ;
			readChar(stream, ')') ;
		}
		else
		{
			result = false ;
		}
	}

	return result ;
}



std::istream & operator>>( std::istream & stream, ColorAssignment & colorAssignment)
{
	colorAssignment.read (stream) ;

	return stream ;
}



}
