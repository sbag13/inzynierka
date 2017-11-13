#ifndef BOUNDARY_AT_LOCATIONS_HPP
#define BOUNDARY_AT_LOCATIONS_HPP



#include <vector>

#include "BoundaryDescription.hpp"
#include "Coordinates.hpp"
#include "NodeLayout.hpp"



namespace microflow
{



class BoundaryAtLocations : public BoundaryDescription
{
	public:


		const std::vector <std::string> & getFileNames() const ;
		void loadLocationFiles (const std::string & directoryPath) ;

		const std::vector <Coordinates> & getNodeLocations() const
		{
			return nodeLocations_ ;
		}


	private:

		virtual bool readElement (const std::string & elementName, 
															std::istream & stream) ;

		void loadCsvFile (const std::string & filePath) ;

		std::vector <std::string> fileNames_ ;
		std::vector <Coordinates> nodeLocations_ ;
} ;



}



#endif
