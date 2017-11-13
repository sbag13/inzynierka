#ifndef LATTICE_ARRANGEMENT_HH
#define LATTICE_ARRANGEMENT_HH



namespace microflow
{



template<unsigned D, unsigned Q>
inline
constexpr unsigned LatticeArrangementBase<D,Q>::
getD()
{
	return D ;
}



template<unsigned D, unsigned Q>
inline
constexpr unsigned LatticeArrangementBase<D,Q>::
getQ()
{
	return Q ;
}



template<unsigned D, unsigned Q>
inline
constexpr Direction::DirectionIndex LatticeArrangementBase<D,Q>::
getIndex( Direction::D direction )
{
	return indexFromDirection_[ static_cast<unsigned>(direction) ] ;
}



template<unsigned D, unsigned Q>
inline
const std::string LatticeArrangementBase<D,Q>::
getName()
{
	return std::string("D") + std::to_string( getD() ) 
									 + "Q"  + std::to_string( getQ() ) ;
}



}
#endif
