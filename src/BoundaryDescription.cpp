#include "BoundaryDescription.hpp"
#include "Logger.hpp"
#include "Exceptions.hpp"
#include "microflowTools.hpp"



using namespace microflow ;
using namespace std ;



void BoundaryDescription::
read (std::istream & stream)
{
	stream >> std::ws ;

	while (true)
	{ // First read checks, if there is data in the stream
		char _c ;
		stream >> std::skipws >> _c ;

		if (stream.eof())
		{
			return ; // FIXME: Throw ? It may be some error.
		}

		if ('#' == _c)
		{
			std::string commentLine ;
			std::getline(stream, commentLine) ;
			continue ;
		}

		if ('{' == _c) break ;

		logger << "Format error, got \'" << _c << "\' (" << (int)_c << ")" ;
		logger << ", expected \'" << '{' << "\'\n" ;
		throw SYNTAX_ERROR_EXCEPTION() ;                                          
	}

	while (true)
	{
		std::string elementName ;
		stream >> elementName ;

		if ("}" == elementName) break ;
		
		if (not readElement (elementName, stream))
		{
			logger << "Wrong field \"" << elementName << "\"\n" ;
			throw SYNTAX_ERROR_EXCEPTION() ;
		}
	}
}



bool BoundaryDescription::
readElement (const string & elementName, istream & stream)
{
	bool result = true ;

	if ("node_type" == elementName)
	{
		readChar(stream, '=') ;

		std::string nodeTypeName ;
		stream >> nodeTypeName ;
		nodeBaseType_ = fromString<NodeBaseType>( nodeTypeName ) ;
	} 
	else if ("velocity" == elementName)
	{
		readChar(stream, '=') ;
		readChar(stream, '(') ;
		stream >> velocity_[0] ;
		readChar(stream, ',') ;
		stream >> velocity_[1] ;
		readChar(stream, ',') ;
		stream >> velocity_[2] ;
		readChar(stream, ')') ;
	}
	else if ("pressure" == elementName)
	{
		readChar(stream, '=') ;
		stream >> pressure_ ;
	}
	else if ("characteristic_length_marker" == elementName)
	{
		readChar(stream, '=') ;
		std::string valueString ;
		stream >> valueString ;

		if ("true" == valueString)
		{
			isCharacteristicLengthMarker_ = true ;
		}
		else if ("false" == valueString)
		{
			isCharacteristicLengthMarker_ = false ;
		}
		else
		{
			logger << "characteristic_length_marker can be only \"true\" or \"false\"" ;
			logger << ", not \"" << valueString << "\"\n" ;
			throw SYNTAX_ERROR_EXCEPTION() ;
		}
	}
	else if ("#" == elementName)
	{
		std::string stmp ;
		std::getline(stream, stmp) ;
	}
	else 
	{
		result = false ;
	}

	return result ;
}


