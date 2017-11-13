#ifndef VTK_TYPES_HPP
#define VTK_TYPES_HPP


#include <vtkType.h>



namespace microflow
{



/*
	Helper class, which "translates" between microflow types and VTK types.
*/
template <class DataType>
class VtkTypes ;



template <>
class VtkTypes <double>
{
	public: 
		typedef vtkDoubleArray ArrayType ;
		static constexpr int VTK_TYPE = VTK_DOUBLE ;
} ;



}



#endif
