#ifndef MODIFICATION_RHO_U_HH
#define MODIFICATION_RHO_U_HH



namespace microflow
{



template <class DataType>
ModificationRhoU::Modification <DataType>::
Modification (Coordinates c, const DataType & v)
: coordinates (c), value (v)
{}



inline void
ModificationRhoU::
addRhoPhysical (Coordinates coordinates, double rho)
{
	rhoPhysical.push_back 
		(
		 ModificationRhoU::Modification<double> (coordinates, rho)
		) ;
}



inline void
ModificationRhoU::
addRhoBoundaryPhysical (Coordinates coordinates, double rho)
{
	rhoBoundaryPhysical.push_back 
		(
		 ModificationRhoU::Modification<double> (coordinates, rho)
		) ;
}



inline void 
ModificationRhoU::
addUPhysical (Coordinates coordinates, double ux, double uy, double uz)
{
	std::array<double,3> uv ;
	uv[0] = ux ;
	uv[1] = uy ;
	uv[2] = uz ;

	uPhysical.push_back
		(
		 ModificationRhoU::Modification <std::array<double,3> > (coordinates, uv)
		) ;
}



inline void 
ModificationRhoU::
addUBoundaryPhysical (Coordinates coordinates, double ux, double uy, double uz)
{
	std::array<double,3> uv ;
	uv[0] = ux ;
	uv[1] = uy ;
	uv[2] = uz ;

	uBoundaryPhysical.push_back
		(
		 ModificationRhoU::Modification <std::array<double,3> > (coordinates, uv)
		) ;
}



inline ModificationRhoU & 
ModificationRhoU::operator += (const ModificationRhoU & right)
{
	uPhysical.insert (uPhysical.end(), right.uPhysical.begin(), right.uPhysical.end()) ;
	rhoPhysical.insert(rhoPhysical.end(), 
										 right.rhoPhysical.begin(), right.rhoPhysical.end()) ;

	rhoBoundaryPhysical.insert(rhoBoundaryPhysical.end(), 
										 right.rhoBoundaryPhysical.begin(), right.rhoBoundaryPhysical.end()) ;
	uBoundaryPhysical.insert(uBoundaryPhysical.end(), 
										 right.uBoundaryPhysical.begin(), right.uBoundaryPhysical.end()) ;

	return *this ;
}



}



#endif
