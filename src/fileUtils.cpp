#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <fstream>
#include <sstream>



#include "fileUtils.hpp"
#include "Exceptions.hpp"



using namespace std ;



namespace microflow
{



void createDirectory( const string directoryPath )
{
	string mkdirCommand = "mkdir -p " + directoryPath ;
	if (0 != system (mkdirCommand.c_str())) 
	{
		THROW( "Can not create directory " + directoryPath ) ;
	}
}



void removeDirectory( const string directoryPath )
{
	string rmdirCommand = "rm -rf " + directoryPath ;
	if (0 != system (rmdirCommand.c_str())) 
	{
		THROW( "Can not remove directory " + directoryPath ) ;
	}
}



void cleanDirectory( const string directoryPath )
{
	string rmCommand = "rm -rf " + directoryPath + "/*" ;
	if (0 != system (rmCommand.c_str())) 
	{
		THROW("Can not clean directory " + directoryPath ) ;
	}
}



// TODO: struct dirent * as argument ???
int extractStepNumberFromVtkFileName( const string vtkFileName, 
																			const string fileNamePattern )
{
	int stepNumber ;
	stringstream ss(vtkFileName) ;
	string btmp = fileNamePattern ;
	ss.read( &(btmp[0]), btmp.size() ) ; // ugly hack !!!

	if (not ss  ||  btmp != fileNamePattern) return -1 ; // not file from microflow
	ss >> stepNumber ;
	ss >> btmp ;
	if (not ss) return -1 ;

	// Allowed vtk file types.
	if (".vti" == btmp  ||  ".vtu" == btmp)
	{
		return stepNumber ;
	}

	return -1 ;
}



vector<string> getFileNamesFromDirectory( const string directoryPath )
{
		struct dirent *entry ;
		DIR *directory ;
		vector<string> fileNames(0) ;

		directory = opendir( directoryPath.c_str() ) ;
		if (directory == NULL) {
			return fileNames ;
		}


		while ( (entry = readdir(directory)) )
		{
			string name( entry->d_name ) ;

			if ( ("." == name) || (".." == name) ) continue ;
			
			fileNames.push_back( name ) ;
		}

	closedir( directory ) ;

	return fileNames ;
}



bool fileExists (const std::string filePath)
{
	std::ifstream infile (filePath) ;
	return infile.good() ;	
}




std::string readFileContents (const std::string & filePath)
{
	std::ifstream f (filePath) ;
	std::stringstream ss ;
	ss << f.rdbuf() ;

	return ss.str() ;
}



/* 
	Look at
	http://stackoverflow.com/questions/51949/how-to-get-file-extension-from-string-in-c
*/
std::string getFileExtension (const string & fileName)
{
	auto found = fileName.find_last_of(".");
	return fileName.substr(found);
}


	
}
