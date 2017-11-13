#ifndef IMAGE_HPP
#define IMAGE_HPP



#include <ostream>
#include <png.hpp>



namespace microflow
{



typedef png::rgb_pixel Pixel ;



bool operator==(const Pixel & p1, const Pixel & p2) ;
std::ostream & operator<<( std::ostream & out, const png::rgb_pixel & pixel ) ;



/*
	Flipped image, normally (0,0) is TOP left
*/
class Image : public png::image<Pixel>
{
	public:

		Image (std::string const & filename) : png::image<Pixel>(filename     ) {}
		Image (char        const * filename) : png::image<Pixel>(filename     ) {}
		Image (size_t width, size_t height)  : png::image<Pixel>(width, height) {}

		Pixel getPixel(unsigned int x, unsigned int y) const ;
		void 	setPixel (size_t x, size_t y, const Pixel & p) ;
		void fill(const Pixel & p) ;
} ;



}



#include "Image.hh"



#endif
