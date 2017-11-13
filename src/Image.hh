#ifndef IMAGE_HH
#define IMAGE_HH



namespace microflow
{



inline
Pixel Image::
getPixel(unsigned int x, unsigned int y) const 
{
	size_t h = png::image< Pixel >::get_height() ;

	return png::image< Pixel >::
		get_pixel( x, h - y - 1 ) ; // flip Y
}



inline
void 	Image::
setPixel (size_t x, size_t y, const Pixel & p)
{
	size_t h = png::image< Pixel >::get_height() ;

	png::image< Pixel >::
	set_pixel( x, h - y - 1, p ) ; // flip Y
}



inline
void Image::
fill(const Pixel & p)
{
	size_t h = png::image< Pixel >::get_height() ;
	size_t w = png::image< Pixel >::get_width () ;

	for (size_t y=0 ; y < h ; y++)
		for (size_t x=0 ; x < w ; x++)
		{
			set_pixel(x, y, p) ;
		}
}



inline
bool operator==(const Pixel & p1, const Pixel & p2)
{
	if (p1.red == p2.red && p1.green == p2.green && p1.blue == p2.blue)
		return true ;
	
	return false ;
}



inline
std::ostream & 
operator<<( std::ostream & out, const png::rgb_pixel & pixel )
{
	out << "{R:" << static_cast<int>( pixel.red ) 
			<< ",G:" << static_cast<int>( pixel.green )
			<< ",B:" << static_cast<int>( pixel.blue )
			<< "}" ;
	return out ;
}



}



#endif
