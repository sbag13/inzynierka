#ifndef EXPANDED_NODE_LAYOUT_HPP
#define EXPANDED_NODE_LAYOUT_HPP



#include "NodeLayout.hpp"
#include "PackedNodeNormalSet.hpp"
#include "SolidNeighborMask.hpp"
#include "Settings.hpp"



namespace microflow
{



/*
	FIXME: This functionality should be embedded in some other class - maybe
				 even in NodeLayout. In fact SolidNeighborMask and NodeNormalsArray
				 are required for computations of boundary nodes.
	FIXME: This should be fixed fast, because a lot of unnecessary code is added
				 to generate additionally ExpandedNodeLayout before creation of
				 TiledLattice objects.
	TODO:  For large geometries it may be faster to classify nodes and compute
				 normal vectors AFTER tiling. This way may also significantly decrease 
				 memory usage for sparse geometries.
*/
class ExpandedNodeLayout
{
	public:

		ExpandedNodeLayout( NodeLayout & nodeLayout ) ;

		Size getSize() const ;

		// Method, which calls all of the below
		void rebuildBoundaryNodes (const Settings & settings) ;

		void classifyNodesPlacedOnBoundary (const Settings & settings) ;
		void classifyPlacementForBoundaryNodes (const Settings & settings) ;

		void computeNormalVectors() ;

		PackedNodeNormalSet getNormalVectors(const Coordinates & coordinates) const ;
		PackedNodeNormalSet getNormalVectors(unsigned x, unsigned y, unsigned z) const ;

		void computeSolidNeighborMasks() ;

		SolidNeighborMask getSolidNeighborMask(const Coordinates & coordinates) const ;
		SolidNeighborMask getSolidNeighborMask(unsigned x, unsigned y, unsigned z) const ;

		bool isNodePlacedOnBoundary (unsigned x, unsigned y, unsigned z) const ;

		const NodeLayout & getNodeLayout() const ;


	private:

		LinearizedMatrix< PackedNodeNormalSet > nodeNormals_ ;
		LinearizedMatrix< SolidNeighborMask   > solidNeighborMasks_ ;

		NodeLayout & nodeLayout_ ;
} ;



}



#include "ExpandedNodeLayout.hh"



#endif
