#ifndef MODIFICATION_RHO_U_HPP
#define MODIFICATION_RHO_U_HPP



#include <vector>
#include <array>

#include "Coordinates.hpp"



namespace microflow
{



class ModificationRhoU
{
	public:

		void addRhoPhysical         (Coordinates coordinates, double rho) ;
		void addRhoBoundaryPhysical (Coordinates coordinates, double rho) ;
		void addUPhysical           (Coordinates coordinates, double ux, double uy, double uz) ;
		void addUBoundaryPhysical   (Coordinates coordinates, double ux, double uy, double uz) ;

		ModificationRhoU & operator += (const ModificationRhoU & right) ;

		template <class DataType>
			class Modification
			{
				public:
					Modification (Coordinates c, const DataType & v) ;

					Coordinates coordinates ;
					DataType   value ;
			} ;

		typedef Modification <double> ModificationRho ;
		typedef Modification <std::array<double,3> > ModificationU ;

		std::vector <ModificationU  > uPhysical ;
		std::vector <ModificationRho> rhoPhysical ;

		std::vector <ModificationU  > uBoundaryPhysical ;
		std::vector <ModificationRho> rhoBoundaryPhysical ;
} ;
	


}



#include "ModificationRhoU.hh"



#endif
