#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP



#include <vector>
#include <string>



namespace microflow
{



void createDirectory( const std::string directoryPath ) ;
void removeDirectory( const std::string directoryPath ) ;
void cleanDirectory ( const std::string directoryPath ) ;

int extractStepNumberFromVtkFileName( const std::string vtkFileName,
																			const std::string fileNamePattern ) ;

std::vector< std::string > 
getFileNamesFromDirectory( const std::string directoryPath ) ;

bool fileExists (const std::string filePath) ;
std::string readFileContents (const std::string & filePath) ;
std::string getFileExtension (const std::string & fileName) ;



}



#endif
