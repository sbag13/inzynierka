#ifndef COLOR_ASSIGNMENT_HPP
#define COLOR_ASSIGNMENT_HPP



#include <istream>
#include <png.hpp>

#include "NodeBaseType.hpp"
#include "BoundaryDescription.hpp"



namespace microflow
{



class ColorAssignment : public BoundaryDescription
{
	public:

		ColorAssignment() ;
		virtual ~ColorAssignment() {} ;

		bool colorEquals( const png::rgb_pixel & color ) const ;

		// TODO: remove and leave only read() method ?
		friend
		std::istream & operator>> (std::istream & stream, 
															 ColorAssignment & colorAssignment ) ;


	private:

		virtual bool readElement (const std::string & elementName,
															std::istream & stream) ;

		png::rgb_pixel color_ ;
} ;


}



#include "ColorAssignment.hh"



#endif
