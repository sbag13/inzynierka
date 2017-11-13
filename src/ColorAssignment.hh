#ifndef COLOR_ASSIGNMENT_HH
#define COLOR_ASSIGNMENT_HH



namespace microflow
{



inline
ColorAssignment::
ColorAssignment() : BoundaryDescription()
{
}



inline
bool ColorAssignment::
colorEquals(const png::rgb_pixel & color) const
{
	return color.red   == color_.red   &&
				 color.green == color_.green &&
	       color.blue  == color_.blue ;
}



}



#endif
