#include "BoundaryAtLocations.hpp"



using namespace std ;
using namespace microflow ;



const vector <string> & BoundaryAtLocations::
getFileNames() const
{
	return fileNames_ ;
}



void BoundaryAtLocations::
loadLocationFiles (const string & directoryPath)
{
	logger << "Loading coordinates\n" ;
	for (auto fName : fileNames_)
	{
		string filePath = directoryPath + "/" + fName ;

		loadCsvFile (filePath) ;
	}
	logger << "Total " << getNodeLocations().size() << " nodes.\n" ;
}



bool BoundaryAtLocations::
readElement (const string & elementName, istream & stream)
{
	bool result = true ;

	if (not BoundaryDescription::readElement (elementName, stream))
	{
		if ("file" == elementName)
		{
			readChar(stream, '=') ;

			string fileName ;
			stream >> fileName ;

			fileNames_.push_back (fileName) ;
		}
		else
		{
			result = false ;
		}
	}

	return result ;
}



void BoundaryAtLocations::
loadCsvFile (const string & filePath)
{
	ifstream file ;

	logger << "Loading \"" << filePath << "\"..." << flush ;

	file.open (filePath) ;
	if (file.fail())
	{
		THROW ("Can not open file \"" + filePath + "\"") ;
	}

	string line ;
	getline (file,line) ;

	const string header = "\"nodeType\",\"Points:0\",\"Points:1\",\"Points:2\"" ;

	if (header != line)
	{
		THROW ("Wrong header in \"" + filePath + "\""
			+ "\nExpected: \"" + header + "\""
			+ "\nGot     : \"" + line + "\""
		) ;
	}

	unsigned nAdded = 0, nSkipped = 0 ;

	while (true)
	{
		short type ;

		file >> skipws >> type ;

		if (file.eof()) break ;

		size_t x,y,z ;
		readChar (file, ',') ;
		file >> x ;
		readChar (file, ',') ;
		file >> y ;
		readChar (file, ',') ;
		file >> z ;

		if (0 != type)
		{
			Coordinates coord (x,y,z) ;
			nodeLocations_.push_back (coord) ;
			nAdded ++ ;
		}
		else
		{
			nSkipped ++ ;
		}
	}

	logger << "OK, " << nAdded << " nodes added, " 
				 << nSkipped << " skipped.\n" ;
}

