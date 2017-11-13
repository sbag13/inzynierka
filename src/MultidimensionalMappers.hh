#ifndef MULTIDIMENSIONAL_MAPPERS_HH
#define MULTIDIMENSIONAL_MAPPERS_HH



namespace microflow
{



HD INLINE 
constexpr unsigned XYZ::
linearize  (unsigned x, unsigned y, unsigned z,
						unsigned width, unsigned height, unsigned depth __attribute__((unused)))
{
	return x  +  y * width  +  z * width * height ;
}



HD INLINE 
constexpr unsigned YXZ::
linearize  (unsigned x, unsigned y, unsigned z,
						unsigned width, unsigned height, unsigned depth __attribute__((unused)))
{
	return y  +  x * height  +  z * height * width ; 
}



HD INLINE 
constexpr unsigned ZigzagNE::
linearize  (unsigned x, unsigned y, unsigned z,
						unsigned width  __attribute__((unused)), 
						unsigned height __attribute__((unused)), 
						unsigned depth  __attribute__((unused)))
{
	return (x + y * 3  + ( (x+1) & 4 ) * (3 - y)) * 2 + (z & 1)  + (z & 2) * 16 ;
}



}



#endif
