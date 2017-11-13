#include "gtest/gtest.h"
#include "NodeCalculator.hpp"
#include "CollisionModels.hpp"
#include "Storage.hpp"



using namespace microflow ;



class TestNode
{
	public:

		typedef double DataTypeType ;

		HD double & rho() 
		{ 
			return rho_ ; 
		}
		HD double & rhoBoundary() 
		{ 
			return rhoBoundary_ ; 
		}

		HD double & u (Axis axis)
		{
			return u_ [static_cast<unsigned>(axis)] ;
		}

		HD double & uBoundary (Axis axis )
		{
			return uBoundary_ [static_cast<unsigned>(axis)] ;
		}

		HD double & f (Direction direction)
		{
			return f (direction.get()) ;
		}
		HD double & f (Direction::D direction)
		{
			return f (D3Q19::getIndex(direction)) ;
		}
		HD double & f (Direction::DirectionIndex directionIndex)
		{
			return f_ [directionIndex] ;
		}

		HD double & fPost (Direction direction)
		{
			return fPost (direction.get()) ;
		}
		HD double & fPost (Direction::D direction)
		{
			return fPost (D3Q19::getIndex(direction)) ;
		}
		HD double & fPost (Direction::DirectionIndex directionIndex)
		{
			return fPost_ [directionIndex] ;
		}

		HD NodeType & nodeType()
		{
			return nodeType_ ;
		}

		HD PackedNodeNormalSet & nodeNormals()
		{
			return nodeNormals_ ;
		}

		HD SolidNeighborMask & solidNeighborMask()
		{
			return solidNeighborMask_ ;
		}
		
	private:

		double rho_ ;
		double u_[3] ;
		double f_[19] ;
		double fPost_[19] ;
		double uBoundary_[3] ;
		double rhoBoundary_ ;

		NodeType nodeType_ ;
		PackedNodeNormalSet nodeNormals_ ;
		SolidNeighborMask solidNeighborMask_ ;
} ;



TEST (NodeCalculator, processBoundaryFluid)
{
#if defined __CUDA_ARCH__
#else

	TestNode node1, node2 ;

	node1.nodeType() = NodeType (NodeBaseType::FLUID) ;
	node2.nodeType() = NodeType (NodeBaseType::FLUID) ;

	for (Direction::DirectionIndex q=0 ; q < 19 ; q++)
	{
		node1.f (q) = q ;
		node2.f (q) = q ;
	}

	double u0LB[3] ;
	u0LB[0] = 5.0 ;
	u0LB[1] = 6.0 ;
	u0LB[2] = 7.0 ;

	NodeCalculator 
	<
		FluidModelIncompressible, 
		CollisionModelBGK,
		D3Q19,
		double,
		StorageOnCPU
	> 
	nodeCalculator (1.0, u0LB, 3.0, 
								#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
									1.0/2.0, 1.0/3.0, 
								#endif
									NodeType (NodeBaseType::BOUNCE_BACK_2), 
									NodeType (NodeBaseType::BOUNCE_BACK_2)
								 ) ;

	nodeCalculator.processBoundary<TestNode,NoOptimizations> (node1) ;
	nodeCalculator.processBoundary<TestNode,UnsafeOptimizations> (node2) ;

	EXPECT_EQ (node1.f (O ), node2.f(O )) ;
	EXPECT_EQ (node1.f (E ), node2.f(E )) ;
	EXPECT_EQ (node1.f (N ), node2.f(N )) ;
	EXPECT_EQ (node1.f (W ), node2.f(W )) ;
	EXPECT_EQ (node1.f (S ), node2.f(S )) ;
	EXPECT_EQ (node1.f (T ), node2.f(T )) ;
	EXPECT_EQ (node1.f (B ), node2.f(B )) ;
	EXPECT_EQ (node1.f (NE), node2.f(NE)) ;
	EXPECT_EQ (node1.f (NW), node2.f(NW)) ;
	EXPECT_EQ (node1.f (SW), node2.f(SW)) ;
	EXPECT_EQ (node1.f (SE), node2.f(SE)) ;
	EXPECT_EQ (node1.f (ET), node2.f(ET)) ;
	EXPECT_EQ (node1.f (EB), node2.f(EB)) ;
	EXPECT_EQ (node1.f (WB), node2.f(WB)) ;
	EXPECT_EQ (node1.f (WT), node2.f(WT)) ;
	EXPECT_EQ (node1.f (NT), node2.f(NT)) ;
	EXPECT_EQ (node1.f (NB), node2.f(NB)) ;
	EXPECT_EQ (node1.f (SB), node2.f(SB)) ;
	EXPECT_EQ (node1.f (ST), node2.f(ST)) ;

	EXPECT_EQ (node1.rho(), node2.rho()) ;
	EXPECT_EQ (node1.u (Axis::X), node2.u (Axis::X)) ;
	EXPECT_EQ (node1.u (Axis::Y), node2.u (Axis::Y)) ;
	EXPECT_EQ (node1.u (Axis::Z), node2.u (Axis::Z)) ;
#endif
}


