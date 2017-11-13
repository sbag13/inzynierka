#ifndef NODE_LAYOUT_HPP
#define NODE_LAYOUT_HPP



#include <cstddef>

#include "LinearizedMatrix.hpp"
#include "ColoredPixelClassificator.hpp"
#include "Image.hpp"
#include "BoundaryDefinitions.hpp"



namespace microflow
{



/*
	TODO: Maybe we should use only some kind of NodeLayoutBitmap, which stores
				only information whether a node is solid or not. Concrete node types
				and placement modifiers may be placed in memory only after tiling. This
				may significantly decrease memory usage for large sparse geometries.
*/
class NodeLayout
{
	public:

		NodeLayout( const ColoredPixelClassificator & coloredPixelClassificator,
								const Image & image, 
								size_t depth) ;
		NodeLayout( const Size & size ) ;

		const NodeType & getNodeType( size_t x, size_t y, size_t z) const ;
		const NodeType & getNodeType( Coordinates coordinates ) const ;

		// Returns SOLID node, if coordinates are outside the geometry
		const NodeType safeGetNodeType( Coordinates coordinates ) const ;

		void setNodeType (size_t x, size_t y, size_t z, NodeType nodeType) ;
		void setNodeType (Coordinates coordinates, NodeType nodeType) ;
		
		Size getSize() const ;
		void resizeWithContent( const Size & newSize ) ;

		bool hasNodeSolidNeighbors( const Coordinates & coordinates ) const ;
		bool isNodePlacedOnBoundary(size_t x, size_t y, size_t z) const ;


		static constexpr const enum NodeBaseType BOUNDARY_MARKER = NodeBaseType::SIZE ;
		void temporaryMarkBoundaryNodes() ;

		void temporaryMarkUndefinedBoundaryNodesAndCovers() ;

		//FIXME: ugly hack to cope with strange structural dependences in RS code.
		//       After testing rewrite completely creation of NodeLayout with 
		//			 simple rules of node replacement. WARNING - it probably will be
		//			 extremely difficult to compare with RS code - only the final result
		//			 will be the same.
		void restoreBoundaryNodes(const ColoredPixelClassificator & coloredPixelClassificator,
															const Image & image ) ;

		typedef LinearizedMatrix< NodeType >::Iterator Iterator ;
		typedef LinearizedMatrix< NodeType >::ConstIterator ConstIterator ;

    Iterator begin() ;
    ConstIterator begin() const ;
    Iterator end() ;
    ConstIterator end() const ;


		const BoundaryDefinitions & getBoundaryDefinitions() const ;
		void setBoundaryDefinitions (BoundaryDefinitions boundaryDefinitions) ;

	
	private:

		LinearizedMatrix< NodeType > nodeTypes_ ;
		BoundaryDefinitions boundaryDefinitions_ ;
} ;



}



#include "NodeLayout.hh"



#endif
