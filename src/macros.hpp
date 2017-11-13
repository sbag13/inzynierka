#ifndef MACROS_HPP
#define MACROS_HPP


#define CONCAT2(x,y)     x ## y
#define CONCAT3(x,y,z)   x ## y ## z
#define CONCAT4(x,y,z,w) x ## y ## z ## w



#define TOSTRING__(symbol) #symbol
#define TO_STRING(symbol) TOSTRING__(symbol)



#endif
