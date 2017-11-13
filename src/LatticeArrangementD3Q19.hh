#ifndef LATTICE_ARRANGEMENT_D3Q19_HH
#define LATTICE_ARRANGEMENT_D3Q19_HH



namespace microflow
{



HD INLINE CONSTEXPR
Direction::DirectionIndex LatticeArrangement<3,19>::
getIndex( Direction::D direction )
{
#ifdef __CUDA_ARCH__  //TODO: avoid code duplication, when there are static constexpr arrays in CUDA
	switch (direction)
	{
		case  0: return 0 ;
		case  1: return 1 ;
		case  3: return 3 ;
		case  4: return 2 ;
		case  5: return 7 + 0 ;
		case  7: return 7 + 4 ;
		case 12: return 4 ;
		case 13: return 7 + 1 ;
		case 15: return 7 + 5 ;
		case 16: return 5 ;
		case 17: return 7 + 2 ;
		case 19: return 7 + 6 ;
		case 20: return 7 + 8 ;
		case 28: return 7 + 10 ;
		case 48: return 6 ;
		case 49: return 7 + 3 ;
		case 51: return 7 + 7 ;
		case 52: return 7 + 9 ;
		case 60: return 7 + 11 ;
		default: return NO_INDEX ;
	}
#else
	return indexFromDirection_[ static_cast<unsigned>(direction) ] ;
#endif
}



HD INLINE CONSTEXPR
Direction::D LatticeArrangement<3,19>::
getC( Direction::DirectionIndex index ) 
{
#ifdef __CUDA_ARCH__  //TODO: avoid code duplication, when there are static constexpr arrays in CUDA
	switch(index)
	{
		case  0: return O ;
		case  1: return E ;
		case  2: return N ;
		case  3: return W ;
		case  4: return S ;
		case  5: return T ;
		case  6: return B ;
		case  7: return NE ;
		case  8: return SE ;
		case  9: return ET ;
		case 10: return EB ;
		case 11: return NW ;
		case 12: return SW ;
		case 13: return WT ;
		case 14: return WB ;
		case 15: return NT ;
		case 16: return NB ;
		case 17: return ST ;
		case 18: return SB ;
		default: return -1 ;
	}
#else
	return c[ index ] ;
#endif
}



HD INLINE CONSTEXPR
double LatticeArrangement<3,19>::
getW( Direction::DirectionIndex index ) 
{
#ifdef __CUDA_ARCH__  //TODO: avoid code duplication, when there are static constexpr arrays in CUDA
	switch(index)
	{
		case  0: return 1.0 / 3.0 ;
		case  1: return 1.0 / 18.0 ;
		case  2: return 1.0 / 18.0 ;
		case  3: return 1.0 / 18.0 ;
		case  4: return 1.0 / 18.0 ;
		case  5: return 1.0 / 18.0 ;
		case  6: return 1.0 / 18.0 ;
		case  7: return 1.0 / 36.0 ;
		case  8: return 1.0 / 36.0 ;
		case  9: return 1.0 / 36.0 ;
		case 10: return 1.0 / 36.0 ;
		case 11: return 1.0 / 36.0 ;
		case 12: return 1.0 / 36.0 ;
		case 13: return 1.0 / 36.0 ;
		case 14: return 1.0 / 36.0 ;
		case 15: return 1.0 / 36.0 ;
		case 16: return 1.0 / 36.0 ;
		case 17: return 1.0 / 36.0 ;
		case 18: return 1.0 / 36.0 ;
		default: return NAN ;
	}
#else
	return w[ index ] ;
#endif
}



}
#endif
