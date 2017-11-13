#ifndef NODE_LAYOUT_WRITER_HPP
#define NODE_LAYOUT_WRITER_HPP



#include "NodeLayout.hpp"

#include <ostream>
#include <string>



namespace microflow
{



class NodeLayoutWriter
{
	public:
		static void saveToVolStream( const NodeLayout & nodeLayout, std::ostream & stream ) ;
		static void saveToVolFile  ( const NodeLayout & nodeLayout, std::string    fileName ) ;
		static void saveToPngFile  ( const NodeLayout & nodeLayout, std::string    fileName ) ;

} ;



}



#endif
