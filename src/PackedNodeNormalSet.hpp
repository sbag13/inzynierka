#ifndef PACKED_NODE_NORMAL_SET_HPP
#define PACKED_NODE_NORMAL_SET_HPP



#include <ostream>

#include "Direction.hpp"
#include "Exceptions.hpp"



namespace microflow
{



class PackedNodeNormalSet
{
	public:

		enum EdgeNodeType {
			CONCAVE_EXTERNAL = 0 ,
			CONVEX_INTERNAL  = 1 ,
			CORNER           = 2 ,
			PARALLEL_WALLS   = 3
		} ;

		HD PackedNodeNormalSet() ;

		HD Direction getNormalVector ( unsigned vectorIndex ) const ;
		   void      addNormalVector ( const Direction & normalVector ) ;
		HD unsigned  getNormalVectorsCounter () const ;
		HD Direction getResultantNormalVector() const ;

		HD enum EdgeNodeType getEdgeNodeType() const ;
		HD void setEdgeNodeType( EdgeNodeType edgeNodeType ) ;


		//TODO: this should be done automatically while adding consecutive normal vectors,
		//       but in this case some coordinates exceed +-1 (for nodes with more than
		//			 3 normal vectors). BTW, investigate this.
		void calculateResultantNormalVector() ;


	private:

		typedef Direction::D  VectorType ;

		HD VectorType extractNormalVector(unsigned vectorIndex) const ;

		unsigned long long normalVectorsCounter_ : 3 ;
		unsigned long long edgeNodeType_         : 2 ;
		unsigned long long normalVectors_        : 7 * (3 * 2) ;
} ;



std::ostream& operator<<(std::ostream& out, const PackedNodeNormalSet & packedNodeNormalSet) ;



}



#include "PackedNodeNormalSet.hh"



#endif
