#ifndef NODE_CALCULATOR_HH
#define NODE_CALCULATOR_HH



#include <cmath>

#include "NodeType.hpp"
#include "Direction.hpp"
#include "Axis.hpp"
#include "Exceptions.hpp"
#include "PackedNodeNormalSet.hpp"
#include "SolidNeighborMask.hpp"
#include "gpuAlgorithms.hh"



namespace microflow
{



#define TEMPLATE_NODE_CALCULATOR_BASE                         \
template<                                                     \
					template<class LatticeArrangement, class DataType>  \
														class FluidModel,                 \
														class LatticeArrangement,         \
														class DataType,                   \
					template<class T> class Storage >                   




#define NODE_CALCULATOR_BASE   \
NodeCalculatorBase< FluidModel, LatticeArrangement, DataType, Storage >



TEMPLATE_NODE_CALCULATOR_BASE
HD NODE_CALCULATOR_BASE::
NodeCalculatorBase( DataType rho0LB, 
								DataType u0LB[ LatticeArrangement::getD()],
								DataType tau 
						 #ifdef ENABLE_UNSAFE_OPTIMIZATIONS
							  , DataType invRho0LB
							  , DataType invTau 
 						 #endif
								)
: CalculatorType( rho0LB, u0LB, tau 
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	, invRho0LB, invTau
#endif
)
{
}



TEMPLATE_NODE_CALCULATOR_BASE
template< class Node >
HD INLINE
void NODE_CALCULATOR_BASE::
initializeAtEquilibrium( Node & node )
{
	NodeType nodeType = node.nodeType() ;

	if ( nodeType.isBoundary() )
	{
		node.rho() = node.rhoBoundary() ;
		for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
		{
			node.u(d) = node.uBoundary(d) ;
		}
	}
	else if ( nodeType.isFluid() )
	{
		node.rho() = CalculatorType::rho0LB_ ;
		for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
		{
			node.u(d) = CalculatorType::u0LB_[d] ;
		}
	}
	else
	{
		node.rho() = NAN ;

		for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
		{
			node.u(d) = NAN ;
		}

		for ( Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++ )
		{
			node.f    ( q ) = NAN ;
			node.fPost( q ) = NAN ;
		}

	}

	if ( nodeType.isBoundary() || nodeType.isFluid() )
	{
		for ( Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
		{
			DataType uArray[LatticeArrangement::getD()] ;
			for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
			{
				uArray[d] = node.u(d) ;
			}

			typedef FluidModel<LatticeArrangement, DataType> FM ;
			node.f(q) = FM::computeFeq( node.rho(), uArray, q ) ;
			node.fPost(q) = node.f(q) ;
		}
	}

	for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
	{
		node.uT0(d) = 0 ;
	}
}



TEMPLATE_NODE_CALCULATOR_BASE
template< class Node, WhereSaveF whereSaveF >
HD INLINE
void NODE_CALCULATOR_BASE::
collideBGK( Node & node )
{
	NodeType nodeType = node.nodeType() ;

	if ( not nodeType.isSolid()  &&   not (nodeType == NodeBaseType::BOUNCE_BACK_2) )
	{
		DataType uArray[LatticeArrangement::getD()] ;
		
#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for (unsigned d=0 ; d < LatticeArrangement::getD() ; d++)
		{
			uArray[d] = node.u(d) ;
		}
		const DataType rho = node.rho() ;

#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
		{
			typedef FluidModel<LatticeArrangement, DataType> FM ;
			DataType feq = FM::computeFeq (rho, uArray, q ) ;

			DataType f_ = node.f(q) ;
			DataType newF ;

#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
			newF = f_ - (f_ - feq) * CalculatorType::invTau_ ;
#else
			newF = f_ - (f_ - feq) / CalculatorType::tau_ ;
#endif

			if (WhereSaveF::F      == whereSaveF) node.f     (q) = newF ;
			if (WhereSaveF::F_POST == whereSaveF) node.fPost (q) = newF ;
		}
	}
}



TEMPLATE_NODE_CALCULATOR_BASE
template <class Node, WhereSaveF whereSaveF>
HD INLINE
void NODE_CALCULATOR_BASE::
collideBounceBack2 (Node & node)
{
	NodeType nodeType = node.nodeType() ;
	
	if (NodeBaseType::BOUNCE_BACK_2 == nodeType.getBaseType())
	{
		for (Direction::DirectionIndex q=0 ; q < LatticeArrangement::getQ() ; q++)
		{
			if (WhereSaveF::F_POST == whereSaveF) node.fPost (q) = node.f (q) ;
		}
	}
}



TEMPLATE_NODE_CALCULATOR_BASE
template< class Node, class Optimization >
HD INLINE
void NODE_CALCULATOR_BASE::
processBoundaryFluid (Node & node)
{
	if (Optimization::shouldEnableUnsafeOptimizations())
	{
		processBoundaryFluid_UnsafeOptimizations (node) ;
	}
	else
	{
		processBoundaryFluid_NoOptimizations (node) ;
	}
}



TEMPLATE_NODE_CALCULATOR_BASE
template< class Node >
HD INLINE
void NODE_CALCULATOR_BASE::
processBoundaryFluid_NoOptimizations (Node & node)
{
	DataType rho = 0.0 ;
	DataType ux  = 0.0 ;
	DataType uy  = 0.0 ;
	DataType uz  = 0.0 ;

	// TODO: RS order only for numeric compatibility, 
	// replace later with something more efficient.
	//for( Direction::DirectionIndex q = 0 ; q < LatticeArrangement::getQ() ; q ++  )

#ifdef __CUDA_ARCH__
	//FIXME: when there are constexpr arrays in CUDA.	
	#pragma unroll
	for (auto d : {O, E, N, W, S, T, B, NE, NW, SW, SE, ET, EB, WB, WT, NT, NB, SB, ST})
#else
	for (auto d : rsDirections)
#endif	
	{
		auto q  =  LatticeArrangement::getIndex( d ) ;
		DataType fq = node.f(q) ;

		rho += fq ;

		Direction direction = LatticeArrangement::getC(q) ;
		ux += direction.getX() * fq ;
		uy += direction.getY() * fq ;
		uz += direction.getZ() * fq ;
	}

	typedef FluidModel<LatticeArrangement, DataType> FM ;

	if (FM::isCompressible)
	{
		ux /= rho ;
		uy /= rho ;
		uz /= rho ;
	}	

	node.u(Axis::X) = ux ;
	node.u(Axis::Y) = uy ;
	node.u(Axis::Z) = uz ;
	node.rho()      = rho ;
}



TEMPLATE_NODE_CALCULATOR_BASE
template< class Node >
HD INLINE
void NODE_CALCULATOR_BASE::
processBoundaryFluid_UnsafeOptimizations (Node & node)
{
	DataType rho = 0.0 ;
	DataType ux  = 0.0 ;
	DataType uy  = 0.0 ;
	DataType uz  = 0.0 ;

	rho += node.f( O  ) ;
	rho += node.f( E  ) ;
	rho += node.f( N  ) ;
	rho += node.f( W  ) ;
	rho += node.f( S  ) ;
	rho += node.f( T  ) ;
	rho += node.f( B  ) ;
	rho += node.f( NE ) ; 
	rho += node.f( NW ) ; 
	rho += node.f( SW ) ; 
	rho += node.f( SE ) ; 
	rho += node.f( ET ) ; 
	rho += node.f( EB ) ; 
	rho += node.f( WB ) ; 
	rho += node.f( WT ) ; 
	rho += node.f( NT ) ; 
	rho += node.f( NB ) ; 
	rho += node.f( SB ) ; 
	rho += node.f( ST ) ;

	ux += node.f( E  ) ;
	ux -= node.f( W  ) ;
	ux += node.f( NE ) ; 
	ux -= node.f( NW ) ; 
	ux -= node.f( SW ) ; 
	ux += node.f( SE ) ; 
	ux += node.f( ET ) ; 
	ux += node.f( EB ) ; 
	ux -= node.f( WB ) ; 
	ux -= node.f( WT ) ; 

	uy += node.f( N  ) ;
	uy -= node.f( S  ) ;
	uy += node.f( NE ) ; 
	uy += node.f( NW ) ; 
	uy -= node.f( SW ) ; 
	uy -= node.f( SE ) ; 
	uy += node.f( NT ) ; 
	uy += node.f( NB ) ; 
	uy -= node.f( SB ) ; 
	uy -= node.f( ST ) ;

	uz += node.f( T  ) ;
	uz -= node.f( B  ) ;
	uz += node.f( ET ) ; 
	uz -= node.f( EB ) ; 
	uz -= node.f( WB ) ; 
	uz += node.f( WT ) ; 
	uz += node.f( NT ) ; 
	uz -= node.f( NB ) ; 
	uz -= node.f( SB ) ; 
	uz += node.f( ST ) ;

	typedef FluidModel<LatticeArrangement, DataType> FM ;

	if (FM::isCompressible)
	{
		ux /= rho ;
		uy /= rho ;
		uz /= rho ;
	}	

	node.u(Axis::X) = ux ;
	node.u(Axis::Y) = uy ;
	node.u(Axis::Z) = uz ;
	node.rho()      = rho ;
}



// Host only version.
TEMPLATE_NODE_CALCULATOR_BASE
template< class Node >
inline 
void NODE_CALCULATOR_BASE::
processBoundaryBounceBack2( Node & node )
{
/*
q=0    O    O
q=1    E    W
q=2    N    S
q=3    W    E		!
q=4    S    N		!
q=5    T    B
q=6    B    T		!
q=7    NE   SW
q=8    SE   NW
q=9    ET   WB
q=10   EB   WT
q=11   NW   SE	!
q=12   SW   NE	!
q=13   WT   EB	!
q=14   WB   ET	!
q=15   NT   SB
q=16   NB   ST	!
q=17   ST   NB
q=18   SB   NT	!
*/

#define f node.f

	#ifdef __CUDA_ARCH__
		#pragma unroll
	#endif	
	for (Direction direction : {E, N, T, NE, SE, ET, EB, NT, ST})
	{
		Direction reverseDirection  =  direction.computeInverse() ;

		microflow::swap (f (direction), f (reverseDirection)) ;
	}
#undef f
}



#ifdef __CUDACC__

template <class DataType>
__device__ void
swapWithTmp (DataType & v1, DataType & v2, volatile DataType & tmp)
{
	tmp = v1 ;
	v1 = v2 ;
	v2 = tmp ;
}



// Device only version.
TEMPLATE_NODE_CALCULATOR_BASE
template <class ThreadMapper, class Node>
DEVICE INLINE 
void NODE_CALCULATOR_BASE::
processBoundaryBounceBack2( Node & node )
{
	constexpr unsigned gX = ThreadMapper::getBlockDimX() ;
	constexpr unsigned gY = ThreadMapper::getBlockDimY() ;
	constexpr unsigned gZ = ThreadMapper::getBlockDimZ() ;

	__shared__ DataType tmpShared [gZ][gY][gX] ;

#define f node.f
	swapWithTmp (f(E) , f(W) , tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
	swapWithTmp (f(N) , f(S) , tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
	swapWithTmp (f(T) , f(B) , tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
	swapWithTmp (f(NE), f(SW), tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
	swapWithTmp (f(SE), f(NW), tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
	swapWithTmp (f(ET), f(WB), tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
	swapWithTmp (f(EB), f(WT), tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
	swapWithTmp (f(NT), f(SB), tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
	swapWithTmp (f(ST), f(NB), tmpShared [threadIdx.z][threadIdx.y][threadIdx.x] ) ;
#undef f
}

#endif



#define TEMPLATE_NODE_CALCULATOR_D3Q19                        \
template<                                                     \
					template<class LatticeArrangement, class DataType>  \
														class FluidModel,                 \
														class CollisionModel,             \
														class DataType,                   \
					template<class T> class Storage >                   




#define NODE_CALCULATOR_D3Q19   \
NodeCalculator< FluidModel, CollisionModel, D3Q19, DataType, Storage >



TEMPLATE_NODE_CALCULATOR_D3Q19
HD INLINE
NODE_CALCULATOR_D3Q19::
NodeCalculator (DataType rho0LB, 
								DataType u0LB[ D3Q19::getD()],
								DataType tau,
						 #ifdef ENABLE_UNSAFE_OPTIMIZATIONS
							  DataType invRho0LB ,
							  DataType invTau ,
						 #endif
								NodeType defaultExternalEdgePressureNode,
								NodeType defaultExternalCornerPressureNode 
							 )
: NodeCalculatorBase< FluidModel, D3Q19, DataType, Storage >
				( rho0LB, u0LB, tau 
			#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
					, invRho0LB, invTau
			#endif
				)
	, defaultExternalEdgePressureNode_ (defaultExternalEdgePressureNode)
	, defaultExternalCornerPressureNode_ (defaultExternalCornerPressureNode)
{
}



TEMPLATE_NODE_CALCULATOR_D3Q19
template< class Node, WhereSaveF whereSaveF >
HD INLINE
void NODE_CALCULATOR_D3Q19::
collide( Node & node )
{
	//FIXME: maybe use an additional internal class template with CollisionModel 
	//				as template parameter ? Look at MEQCalulator. In this case remove 
	//				CollisionModel::isBGK attributes.
	if ( CollisionModel::isBGK )
	{
		BaseCalculator::template collideBGK <Node, whereSaveF> (node) ;
	}
	else if ( CollisionModel::isMRT )
	{
		collideMRT <Node, whereSaveF> (node) ;
	}
	else
	{
		THROW( "UNIMPLEMENTED" ) ;
	}

	BaseCalculator::template collideBounceBack2 <Node, whereSaveF> (node) ;
}



//FIXME: Comments in Polish.
TEMPLATE_NODE_CALCULATOR_D3Q19
template <class Node, WhereSaveF whereSaveF>
HD INLINE
void NODE_CALCULATOR_D3Q19::
collideMRT( Node & node )
{
	// Macierz wspolczynnikow do konwersji predkosci do przestrzeni pedu i na odwrot
// Order of coefficients changed to fit LatticeArrangement::c. Numbers in comments are
// indices in RS version.
// In the current version of code M array is the same as in RS, because for numeric 
// compatibility the order of f_i accumulation must be the same.
// In this way we may also use s[] and and d[] arrays from RS code.
	const DataType M[ 19 ][ 19 ]=	
	{
//         0    1    2    3    4    5    6   7   8   9  10  11  12  13  14  15  16  17  18   
/* 00 */{  1,   1,   1,   1,   1,   1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
/* 01 */{-30, -11, -11, -11, -11, -11, -11,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8},
/* 02 */{ 12,  -4,  -4,  -4,  -4,  -4,  -4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
/* 03 */{  0,   1,   0,  -1,   0,   0,   0,  1, -1, -1,  1,  1,  1, -1, -1,  0,  0,  0,  0},
/* 04 */{  0,  -4,   0,   4,   0,   0,   0,  1, -1, -1,  1,  1,  1, -1, -1,  0,  0,  0,  0},
/* 05 */{  0,   0,   1,   0,  -1,   0,   0,  1,  1, -1, -1,  0,  0,  0,  0,  1,  1, -1, -1},
/* 06 */{  0,   0,  -4,   0,   4,   0,   0,  1,  1, -1, -1,  0,  0,  0,  0,  1,  1, -1, -1},
/* 07 */{  0,   0,   0,   0,   0,   1,  -1,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1},
/* 08 */{  0,   0,   0,   0,   0,  -4,   4,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1},
/* 09 */{  0,   2,  -1,   2,  -1,  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2},
/* 10 */{  0,  -4,   2,  -4,   2,   2,   2,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2},
/* 11 */{  0,   0,   1,   0,   1,  -1,  -1,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0},
/* 12 */{  0,   0,  -2,   0,  -2,   2,   2,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0},
/* 13 */{  0,   0,   0,   0,   0,   0,   0,  1, -1,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0},
/* 14 */{  0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  1, -1},
/* 15 */{  0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  1, -1,  1, -1,  0,  0,  0,  0},
/* 16 */{  0,   0,   0,   0,   0,   0,   0,  1, -1, -1,  1, -1, -1,  1,  1,  0,  0,  0,  0},
/* 17 */{  0,   0,   0,   0,   0,   0,   0, -1, -1,  1,  1,  0,  0,  0,  0,  1,  1, -1, -1},
/* 18 */{  0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  1, -1, -1,  1, -1,  1,  1, -1}
	};
//0  1  2  3  4  5  6  7  11 	12 	8  9  10 	14 	13 	15 	16 	18 	17 


#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	const DataType dInv[ 19 ]={1/19.0, 1/2394.0, 1/252.0, 1/10.0, 1/40.0, 1/10.0, 1/40.0, 1/10.0, 1/40.0, 1/36.0, 1/72.0, 1/12.0, 1/24.0, 1/4.0, 1/4.0, 1/4.0, 1/8.0, 1/8.0, 1/8.0}; //D=M*MT
#else
	const DataType d[ 19 ]={19.0,2394.0,252.0,10.0,40.0,10.0,40.0,10.0,40.0,36.0,72.0,12.0,24.0,4.0,4.0,4.0,8.0,8.0,8.0}; //D=M*MT
#endif
	
	DataType meq ;
	DataType m[ 19 ];
	DataType s[ 19 ];
	bool isSEmpty [19] ;

#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	#define INV_TAU BaseCalculator::CalculatorType::invTau_
#else
	#define INV_TAU (1.0/BaseCalculator::CalculatorType::tau_)
#endif
	//Czestotliwosci relaksacji dla modelu MRT wg. D.D'Humieres et al 2002
		isSEmpty[0] = true  ; s[0]=0.0;  // rho (conserved)
		isSEmpty[1] = false ; s[1]=1.19; //wynika ze wzoru na lepkosci objetosciowa przyjmowane 1.19
		isSEmpty[2] = false ; s[2]=1.4;
		isSEmpty[3] = true  ; s[3]=0.0;  // rho*ux (conserved)
		isSEmpty[4] = false ; s[4]=1.2;
		isSEmpty[5] = true  ; s[5]=0.0;  // rho*uy (conserved)
		isSEmpty[6] = false ; s[6]=1.2;  // =s[4]
		isSEmpty[7] = true  ; s[7]=0.0;  // rho*uz (conserved)
		isSEmpty[8] = false ; s[8]=1.2;  // =s[4]
		isSEmpty[9] = false ; s[9]= INV_TAU ;
		isSEmpty[10] = false; s[10]=1.4;
		isSEmpty[11] = false; s[11]= INV_TAU ;
		isSEmpty[12] = false; s[12]=1.4;
		isSEmpty[13] = false; s[13]= INV_TAU ;
		isSEmpty[14] = false; s[14]= INV_TAU ;
		isSEmpty[15] = false; s[15]= INV_TAU ;
		isSEmpty[16] = false; s[16]=1.2; //1.98;
		isSEmpty[17] = false; s[17]=1.2; //1.98;
		isSEmpty[18] = false; s[18]=1.2; //1.98;
#undef INV_TAU

	NodeType nodeType = node.nodeType() ;

	if ( not nodeType.isSolid()  &&   not (nodeType == NodeBaseType::BOUNCE_BACK_2) )
	{

		//Transformacja z przestrzeni predkosci do przestrzeni pedu
#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for( unsigned k=0 ; k < 19 ; k++ )
		{
			m[k]=0.0;

			unsigned nRS = 0 ;
			//FIXME: when there are constexpr arrays in CUDA: use	rsDirections[].
#ifdef __CUDA_ARCH__
			#pragma unroll
#endif
			for (auto d : {O, E, N, W, S, T, B, NE, NW, SW, SE, ET, EB, WB, WT, NT, NB, SB, ST})
			{
				auto n  =  D3Q19::getIndex( d ) ;
				m[k] += node.f(n) * M[k][nRS]; //M
				nRS++ ;
			}
		}

		DataType rho = node.rho() ;
		DataType uX = node.u(X) ;
		DataType uY = node.u(Y) ;
		DataType uZ = node.u(Z) ;
#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for ( unsigned k=0 ; k < 19 ; k++ )
		{
			if (not isSEmpty [k]) //FIXME: probably not needed.
			{
				meq = MEQCalculator<FluidModel,0>::computeMEQ
							(
								rho,
								uX, uY, uZ,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS	
								BaseCalculator::CalculatorType::invRho0LB_,
#else
								BaseCalculator::CalculatorType::rho0LB_,
#endif
								k
							) ;
				m[k] = m[k] - s[k] * ( m[k] - meq ) ;
			}
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
			m [k] *= dInv [k] ;
#else
			m[k] /= d[k] ;
#endif
		}
	

		// Transformacja wynikow z powrotem do przestrzeni predkosci
		unsigned kRS = 0 ;
#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for (auto d : {O, E, N, W, S, T, B, NE, NW, SW, SE, ET, EB, WB, WT, NT, NB, SB, ST})
		{
			DataType fPost = 0.0 ;

#ifdef __CUDA_ARCH__
			#pragma unroll
#endif
			for(unsigned n=0 ; n < 19 ; n++ )
			{
				fPost += m[n] * M[n][kRS] ; //MT (M transponowana)
			}
			kRS++ ;

			auto k  =  D3Q19::getIndex( d ) ;

			if (WhereSaveF::F_POST == whereSaveF) node.fPost(k) = fPost ;
			if (WhereSaveF::F      == whereSaveF) node.f    (k) = fPost ;
		}
	}
}


/*
	WARNING: in computeMEQ() methods the parameter k is not an DirectionIndex, 
					 but the same number, as in RS code.
*/
TEMPLATE_NODE_CALCULATOR_D3Q19
template< int Dummy >
HD INLINE
DataType NODE_CALCULATOR_D3Q19::MEQCalculator< FluidModelIncompressible, Dummy>::
computeMEQ
( 
 DataType rho, 
 DataType u, DataType v, DataType w,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
 DataType invRho0LB,
#else
 DataType rho0_LB,
#endif
 int k 
)
{
	DataType x = 0 ;

	//Odpowiedniki LBGK
	const DataType alpha = 3.0 ;
	const DataType beta  = -11.0/2.0 ;
	const DataType gamma = -0.5 ;

	// Zoptymalizowane parametry wg D. D'Humieres et al 2002
	//alpha=0.0;
	//beta=-475.0/63.0;
	//gamma=0.0;

#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
	#define APPLY_RHO0LB *invRho0LB
#else
	#define APPLY_RHO0LB /rho0_LB
#endif
	switch (k)
	{
		case 0: x = rho; 
		break ;
		case 1: x = -11.0 * rho + 19.0 * rho*rho * ( u*u + v*v + w*w ) APPLY_RHO0LB ;
		break ;
		case 2: x = rho * alpha + beta * rho*rho * (u*u + v*v + w*w ) APPLY_RHO0LB ;
		break ;
		case 3: x = rho * u ;
		break ;
		case 4:	x = -2.0/3.0 * rho * u ;
		break ;
		case 5: x = rho * v ;
		break ;
		case 6: x = -2.0/3.0 * rho * v ;
		break ;
		case 7: x = rho * w ;
		break ;
		case 8: x = -2.0/3.0 * rho * w ;
		break ;
		case 9: x = rho*rho * ( 2.0 * u*u - ( v*v + w*w ) )   APPLY_RHO0LB ;
		break ;
		case 10: x = gamma * rho*rho * ( 2.0 * u*u - ( v*v + w*w ) )   APPLY_RHO0LB ;
		break ;
		case 11: x = rho*rho * ( v*v - w*w )   APPLY_RHO0LB ;
		break ;
		case 12: x = gamma * rho*rho * ( v*v - w*w )   APPLY_RHO0LB ;
		break ;
		case 13: x = rho*rho * u * v   APPLY_RHO0LB ;
		break ;
		case 14: x = rho*rho * v * w   APPLY_RHO0LB ;
		break ;
		case 15: x = rho*rho * u * w   APPLY_RHO0LB ;
		break ;
		case 16: x = 0.0 ;
		break ;
		case 17: x = 0.0 ;
		break ;
		case 18: x = 0.0 ;
		break ;
		default: x = 0.0 ;
		break ;
	}
#undef APPLY_RHO0LB

	return x ;
}



TEMPLATE_NODE_CALCULATOR_D3Q19
template< int Dummy >
HD INLINE
DataType NODE_CALCULATOR_D3Q19::MEQCalculator< FluidModelQuasicompressible, Dummy>::
computeMEQ
( 
 DataType rho, 
 DataType u, DataType v, DataType w,
#ifdef ENABLE_UNSAFE_OPTIMIZATIONS
 DataType invRho0LB,
#else
 DataType rho0_LB,
#endif
 int k 
)
{
	DataType x;

	//Odpowiedniki LBGK
	const DataType alpha = 3.0 ;
	const DataType beta  = -11.0/2.0 ;
	const DataType gamma = -0.5 ;

	// Zoptymalizowane parametry wg D. D'Humieres et al 2002
	//alpha=0.0;
	//beta=-475.0/63.0;
	//gamma=0.0;

	switch (k)
	{
		case 0: x = rho; 
		break ;
		case 1: x = -11.0 * rho + 19.0 * rho * ( u*u + v*v + w*w ) ;
		break ;
		case 2: x = rho * alpha + beta * rho * (u*u + v*v + w*w ) ;
		break ;
		case 3: x = rho * u ;
		break ;
		case 4:	x = -2.0/3.0 * rho * u ;
		break ;
		case 5: x = rho * v ;
		break ;
		case 6: x = -2.0/3.0 * rho * v ;
		break ;
		case 7: x = rho * w ;
		break ;
		case 8: x = -2.0/3.0 * rho * w ;
		break ;
		case 9: x = rho * ( 2.0 * u*u - ( v*v + w*w ) ) ;
		break ;
		case 10: x = gamma * rho * ( 2.0 * u*u - ( v*v + w*w ) ) ;
		break ;
		case 11: x = rho * ( v*v - w*w ) ;
		break ;
		case 12: x = gamma * rho * ( v*v - w*w ) ;
		break ;
		case 13: x = rho * ( u * v ) ;
		break ;              
		case 14: x = rho * ( v * w ) ;
		break ;              
		case 15: x = rho * ( u * w ) ;
		break ;
		case 16: x = 0.0 ;
		break ;
		case 17: x = 0.0 ;
		break ;
		case 18: x = 0.0 ;
		break ;
		default: x = 0.0 ;
		break ;
	}

	return x ;
}



/* 
	RS numbers for f_i functions:

	0   O
	1   E
	2   N
	3   W
	4   S
	5   T
	6   B
	7   NE
	8   NW
	9   SW
	10  SE
	11  ET
	12  EB
	13  WB
	14  WT
	15  NT
	16  NB
	17  SB
	18  ST
 */
TEMPLATE_NODE_CALCULATOR_D3Q19
template 
<
#ifdef __CUDA_ARCH__
	class ThreadMapper,
#endif
	class Node, 
	class Optimization
>
HD INLINE
void NODE_CALCULATOR_D3Q19::
processBoundary( Node & node )
{
	auto nodeBaseType  =  node.nodeType().getBaseType()  ;
	auto placementModifier  =  node.nodeType().getPlacementModifier()  ;
	
	#define f node.f
	#define fPost node.fPost
	#define ux node.u(Axis::X)
	#define uy node.u(Axis::Y)
	#define uz node.u(Axis::Z)
	#define uxB node.uBoundary(Axis::X)
	#define uyB node.uBoundary(Axis::Y)
	#define uzB node.uBoundary(Axis::Z)
	#define rho node.rho()
	#define rhoB node.rhoBoundary()

	DataType N_x, N_y, N_z ;

	typedef FluidModel<D3Q19, DataType> FM ;


	switch (nodeBaseType)
	{
		case NodeBaseType::SOLID:
		break ;

		case NodeBaseType::FLUID:

			BaseCalculator::template processBoundaryFluid <Node,Optimization> (node) ;

		break ;


		case NodeBaseType::BOUNCE_BACK_2:

			#ifdef __CUDA_ARCH__
				BaseCalculator::template processBoundaryBounceBack2 <ThreadMapper, Node>
					(node) ;
			#else
				BaseCalculator::processBoundaryBounceBack2 (node) ;
			#endif

		break ;


		case NodeBaseType::VELOCITY:

			rho = NAN ;

			switch (placementModifier)
			{
				case PlacementModifier::NORTH:

					if (FM::isCompressible)
					{
						rho = (f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
								f(T) + f(B) + 2.0 * (f(NT) + f(NB) + f(N) + f(NE) + f(NW)))/(1.0 + uyB) ;

						N_x = 0.5 * ( f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W) ) -
									1.0/3.0 * uxB * rho ;
						N_z = 0.5 * ( f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B) ) -
									1.0/3.0 * uzB * rho ;

						f(S) = f(N) - 1.0/3.0 * uyB * rho ;
						f(SW) = f(NE) + N_x - 1.0/6.0 * (uxB + uyB) * rho ;
						f(SE) = f(NW) - N_x + 1.0/6.0 * (uxB - uyB) * rho ;
						f(SB) = f(NT) + N_z - 1.0/6.0 * (uyB + uzB) * rho ;
						f(ST) = f(NB) - N_z + 1.0/6.0 * ( - uyB + uzB) * rho ;
					}
					if (FM::isIncompressible)
					{
						rho =  - uyB + ( f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
								f(T) + f(B) + 2.0 * ( f(NT) + f(NB) + f(N) + f(NE) + f(NW) ) ) ;

						N_x = 0.5 * (f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W)) - 1.0/3.0 * uxB ;
						N_z = 0.5 * (f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B)) - 1.0/3.0 * uzB ;

						f(S) = f(N) - 1.0/3.0 * uyB ;
						f(SW) = f(NE) + N_x - 1.0/6.0 * (uxB + uyB) ;
						f(SE) = f(NW) - N_x + 1.0/6.0 * (uxB - uyB) ;
						f(SB) = f(NT) + N_z - 1.0/6.0 * (uyB + uzB) ;
						f(ST) = f(NB) - N_z + 1.0/6.0 * ( - uyB + uzB) ;
					}
				break  ;

				case PlacementModifier::SOUTH:

					if (FM::isCompressible)
					{
						rho = (f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
								f(T) + f(B) + 2.0 * (f(SE) + f(SB) + f(ST) + f(S) + f(SW)))/(1.0 - uyB) ;

						N_x = 0.5 * ( f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W) ) - 
									1.0/3.0 * uxB * rho ;
						N_z = 0.5 * ( f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B) ) - 
									1.0/3.0 * uzB * rho ;

						f(N) = f(S) + 1.0/3.0 * uyB * rho ;
						f(NE) = f(SW) - N_x + 1.0/6.0 * (uxB + uyB) * rho ;
						f(NW) = f(SE) + N_x + 1.0/6.0 * ( - uxB + uyB) * rho ;
						f(NT) = f(SB) - N_z + 1.0/6.0 * (uyB + uzB) * rho ;
						f(NB) = f(ST) + N_z + 1.0/6.0 * (uyB - uzB) * rho ;
					}
					if (FM::isIncompressible)
					{
						rho = uyB + ( f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
								f(T) + f(B) + 2.0 * ( f(SE) + f(SB) + f(ST) + f(S) + f(SW) ) ) ;

						N_x = 0.5 * (f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W)) - 1.0/3.0 * uxB ;
						N_z = 0.5 * (f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B)) - 1.0/3.0 * uzB ;

						f(N) = f(S) + 1.0/3.0 * uyB ;
						f(NE) = f(SW) - N_x + 1.0/6.0 * (uxB + uyB) ;
						f(NW) = f(SE) + N_x + 1.0/6.0 * ( - uxB + uyB) ;
						f(NT) = f(SB) - N_z + 1.0/6.0 * (uyB + uzB) ;
						f(NB) = f(ST) + N_z + 1.0/6.0 * (uyB - uzB) ;
					}
				break  ;

				case PlacementModifier::EAST:

					if (FM::isCompressible)
					{
						rho = (f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
								f(T) + f(B) + 2.0 * (f(E) + f(SE) + f(ET) + f(EB) + f(NE)))/(1.0 + uxB) ;

						N_y = 0.5 * ( f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S) ) - 
									1.0/3.0 * uyB * rho ;
						N_z = 0.5 * ( f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B) ) - 
									1.0/3.0 * uzB * rho ;

						f(W) = f(E) - 1.0/3.0 * uxB * rho ;
						f(WB) = f(ET) + N_z - 1.0/6.0 * (uxB + uzB) * rho ;
						f(WT) = f(EB) - N_z + 1.0/6.0 * ( - uxB + uzB) * rho ;
						f(SW) = f(NE) + N_y - 1.0/6.0 * (uxB + uyB) * rho ;
						f(NW) = f(SE) - N_y + 1.0/6.0 * ( - uxB + uyB) * rho ;
					}
					if (FM::isIncompressible)
					{
						rho =  - uxB + ( f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
								f(T) + f(B) + 2.0 * (f(E) + f(SE) + f(ET) + f(EB) + f(NE) ) ) ;

						N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) - 1.0/3.0 * uyB ;
						N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) - 1.0/3.0 * uzB ;

						f(W) = f(E) - 1.0/3.0 * uxB ;
						f(WB) = f(ET) + N_z - 1.0/6.0 * (uxB + uzB) ;
						f(WT) = f(EB) - N_z + 1.0/6.0 * ( - uxB + uzB) ;
						f(SW) = f(NE) + N_y - 1.0/6.0 * (uxB + uyB) ;
						f(NW) = f(SE) - N_y + 1.0/6.0 * ( - uxB + uyB) ;
					}
				break  ;

				case PlacementModifier::WEST: 

					if (FM::isCompressible)
					{
						rho = (f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
								f(T) + f(B) + 2.0 * (f(WB) + f(WT) + f(W) + f(NW) + f(SW)))/(1.0 - uxB) ;

						N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) - 
									1.0/3.0 * uyB * rho ;
						N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) - 
									1.0/3.0 * uzB * rho ;

						f(E) = f(W) + 1.0/3.0 * uxB * rho ;
						f(ET) = f(WB) - N_z + 1.0/6.0 * (uxB + uzB) * rho ;
						f(EB) = f(WT) + N_z + 1.0/6.0 * (uxB - uzB) * rho ;
						f(NE) = f(SW) - N_y + 1.0/6.0 * (uxB + uyB) * rho ;
						f(SE) = f(NW) + N_y + 1.0/6.0 * (uxB - uyB) * rho ;
					}
					if (FM::isIncompressible)
					{
						rho = uxB + (f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
								f(T) + f(B) + 2.0 * (f(WB) + f(WT) + f(W) + f(NW) + f(SW))) ;

						N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) - 1.0/3.0 * uyB ;
						N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) - 1.0/3.0 * uzB ;

						f(E) = f(W) + 1.0/3.0 * uxB ;
						f(ET) = f(WB) - N_z + 1.0/6.0 * (uxB + uzB) ;
						f(EB) = f(WT) + N_z + 1.0/6.0 * (uxB - uzB) ;
						f(NE) = f(SW) - N_y + 1.0/6.0 * (uxB + uyB) ;
						f(SE) = f(NW) + N_y + 1.0/6.0 * (uxB - uyB) ;
					}
				break  ;
				
				case PlacementModifier::BOTTOM:
					
					if (FM::isCompressible)
					{
						rho = (f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
								f(NW) + f(SW) + 2.0 * (f(EB) + f(WB) + f(NB) + f(SB) + f(B)))/(1.0 - uzB) ;

						N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) - 
									1.0/3.0 * uxB * rho ;
						N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) - 
									1.0/3.0 * uyB * rho ;

						f(T) = f(B) + 1.0/3.0 * uzB * rho ;
						f(ET) = f(WB) - N_x + 1.0/6.0 * (uxB + uzB) * rho ;
						f(ST) = f(NB) + N_y + 1.0/6.0 * ( - uyB + uzB) * rho ;
						f(NT) = f(SB) - N_y + 1.0/6.0 * (uyB + uzB) * rho ;
						f(WT) = f(EB) + N_x + 1.0/6.0 * ( - uxB + uzB) * rho ;
					}
					if (FM::isIncompressible)
					{
						rho = uzB + (f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
								f(NW) + f(SW) + 2.0 * (f(EB) + f(WB) + f(NB) + f(SB) + f(B))) ;

						N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) - 
									1.0/3.0 * uxB ;
						N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) - 
									1.0/3.0 * uyB ;

						f(T) = f(B) + 1.0/3.0 * uzB ;
						f(ET) = f(WB) - N_x + 1.0/6.0 * (uxB + uzB) ;
						f(ST) = f(NB) + N_y + 1.0/6.0 * ( - uyB + uzB) ;
						f(NT) = f(SB) - N_y + 1.0/6.0 * (uyB + uzB) ;
						f(WT) = f(EB) + N_x + 1.0/6.0 * ( - uxB + uzB) ;
					}
				break  ;

				case PlacementModifier::TOP:

					if (FM::isCompressible)
					{
						rho = (f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
								f(NW) + f(SW) + 2.0 * (f(ET) + f(WT) + f(NT) + f(ST) + f(T)))/(1.0 + uzB) ;

						N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) - 
									1.0/3.0 * uxB * rho ;
						N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) - 
									1.0/3.0 * uyB * rho ;

						f(B) = f(T) - 1.0/3.0 * uzB * rho ;
						f(WB) = f(ET) + N_x - 1.0/6.0 * (uxB + uzB) * rho ;
						f(NB) = f(ST) - N_y + 1.0/6.0 * (uyB - uzB) * rho ;
						f(SB) = f(NT) + N_y - 1.0/6.0 * (uyB + uzB) * rho ;
						f(EB) = f(WT) - N_x + 1.0/6.0 * (uxB - uzB) * rho ;
					}
					if (FM::isIncompressible)
					{
						rho =  - uzB + (f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
								f(NW) + f(SW) + 2.0 * (f(ET) + f(WT) + f(NT) + f(ST) + f(T))) ;

						N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) - 1.0/3.0 * uxB ;
						N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) - 1.0/3.0 * uyB ;

						f(B) = f(T) - 1.0/3.0 * uzB ;
						f(WB) = f(ET) + N_x - 1.0/6.0 * (uxB + uzB) ;
						f(NB) = f(ST) - N_y + 1.0/6.0 * (uyB - uzB) ;
						f(SB) = f(NT) + N_y - 1.0/6.0 * (uyB + uzB) ;
						f(EB) = f(WT) - N_x + 1.0/6.0 * (uxB - uzB) ;
					}
				break  ;

				case PlacementModifier::EXTERNAL_EDGE                            :
				case PlacementModifier::EXTERNAL_EDGE_PRESSURE_TANGENTIAL        : 
				case PlacementModifier::INTERNAL_EDGE                            : 
				case PlacementModifier::EXTERNAL_CORNER                          : 
				case PlacementModifier::EXTERNAL_CORNER_PRESSURE_TANGENTIAL      : 
				case PlacementModifier::CORNER_ON_EDGE_AND_PERPENDICULAR_PLANE   : 

					THROW ("Not implemented") ;

				break  ;

				default:
					THROW ("Undefined placementModifier for VELOCITY node") ;
			}
		break  ;


		case NodeBaseType::VELOCITY_0:

			ux = 0.0 ;
			uy = 0.0 ;
			uz = 0.0 ;

			switch (placementModifier)
			{

				case PlacementModifier::NORTH : 
										
					rho = (f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
							f(T) + f(B) + 2.0 * (f(NT) + f(NB) + f(N) + f(NE) + f(NW))) ;

					N_x = 0.5 * (f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W)) ;
					N_z = 0.5 * (f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B)) ;

					f(S) = f(N) ;
					f(SW) = f(NE) + N_x ;
					f(SE) = f(NW) - N_x ;
					f(SB) = f(NT) + N_z ;
					f(ST) = f(NB) - N_z ;
				break  ;

				case PlacementModifier::SOUTH : 

					rho = f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
						f(T) + f(B) + 2.0 * (f(SE) + f(SB) + f(ST) + f(S) + f(SW)) ;

					N_x = 0.5 * (f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W)) ;
					N_z = 0.5 * (f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B)) ;

					f(N) = f(S) ;
					f(NE) = f(SW) - N_x ;
					f(NW) = f(SE) + N_x ;
					f(NT) = f(SB) - N_z ;
					f(NB) = f(ST) + N_z ;
				break  ;

				case PlacementModifier::EAST  : 

					rho = f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
						f(T) + f(B) + 2.0 * (f(E) + f(SE) + f(ET) + f(EB) + f(NE)) ;

					N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) ;
					N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) ;

					f(W) = f(E) ;
					f(WB) = f(ET) + N_z ;
					f(WT) = f(EB) - N_z ;
					f(SW) = f(NE) + N_y ;
					f(NW) = f(SE) - N_y ;
				break  ;
				
				case PlacementModifier::WEST  : 

					rho = f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
						f(T) + f(B) + 2.0 * (f(WB) + f(WT) + f(W) + f(NW) + f(SW)) ;

					N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) ;
					N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) ;

					f(E) = f(W) ;
					f(ET) = f(WB) - N_z ;
					f(EB) = f(WT) + N_z ;
					f(NE) = f(SW) - N_y ;
					f(SE) = f(NW) + N_y ;
				break  ;

				case PlacementModifier::BOTTOM: 

					rho = f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
						f(NW) + f(SW) + 2.0 * (f(EB) + f(WB) + f(NB) + f(SB) + f(B)) ;

					N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) ;
					N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) ;

					f(T) = f(B) ;
					f(ET) = f(WB) - N_x ;
					f(ST) = f(NB) + N_y ;
					f(NT) = f(SB) - N_y ;
					f(WT) = f(EB) + N_x ;
				break  ;

				case PlacementModifier::TOP: 

					rho = f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
						f(NW) + f(SW) + 2.0 * (f(ET) + f(WT) + f(NT) + f(ST) + f(T)) ;

					N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) ;
					N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) ;

					f(B) = f(T) ;
					f(WB) = f(ET) + N_x ;
					f(NB) = f(ST) - N_y ;
					f(SB) = f(NT) + N_y ;
					f(EB) = f(WT) - N_x ;
				break  ;

				default:
					THROW ("Undefined placementModifier for VELOCITY_0 node") ;
			}
		break  ;



		case NodeBaseType::PRESSURE:   

		rho  =  rhoB  ;
		ux   =  0.0  ;
		uy   =  0.0  ;
		uz   =  0.0  ;

			switch (placementModifier)
			{
				case PlacementModifier::NORTH:

					if (FM::isCompressible)
					{
						uy =  - 1.0 + (f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
								f(T) + f(B) + 2.0 * (f(NT) + f(NB) + f(N) + f(NE) + f(NW)))/rho ;

						N_x = 0.5 * (f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W)) - 
									1.0/3.0 * ux * rho ;
						N_z = 0.5 * (f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B)) - 
									1.0/3.0 * uz * rho ;

						f(S) = f(N) - 1.0/3.0 * uy * rho ;
						f(SW) = f(NE) + N_x - 1.0/6.0 * (ux + uy) * rho ;
						f(SE) = f(NW) - N_x + 1.0/6.0 * (ux - uy) * rho ;
						f(SB) = f(NT) + N_z - 1.0/6.0 * (uy + uz) * rho ;
						f(ST) = f(NB) - N_z + 1.0/6.0 * ( - uy + uz) * rho ;
					}
					if (FM::isIncompressible)
					{
						uy =  - rho + (f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
								f(T) + f(B) + 2.0 * (f(NT) + f(NB) + f(N) + f(NE) + f(NW))) ;

						N_x = 0.5 * (f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W)) - 1.0/3.0 * ux ;
						N_z = 0.5 * (f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B)) - 1.0/3.0 * uz ;

						f(S) = f(N) - 1.0/3.0 * uy ;
						f(SW) = f(NE) + N_x - 1.0/6.0 * (ux + uy) ;
						f(SE) = f(NW) - N_x + 1.0/6.0 * (ux - uy) ;
						f(SB) = f(NT) + N_z - 1.0/6.0 * (uy + uz) ;
						f(ST) = f(NB) - N_z + 1.0/6.0 * ( - uy + uz) ;
					}
				break  ;

				case PlacementModifier::SOUTH : 
					
					if (FM::isCompressible)
					{
						uy = 1.0 - (f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
								f(T) + f(B) + 2.0 * (f(SE) + f(SB) + f(ST) + f(S) + f(SW)))/rho ;

						N_x = 0.5 * (f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W)) - 
									1.0/3.0 * ux * rho ;
						N_z = 0.5 * (f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B)) - 
									1.0/3.0 * uz * rho ;

						f(N) = f(S) + 1.0/3.0 * uy * rho ;
						f(NE) = f(SW) - N_x + 1.0/6.0 * (ux + uy) * rho ;
						f(NW) = f(SE) + N_x + 1.0/6.0 * ( - ux + uy) * rho ;
						f(NT) = f(SB) - N_z + 1.0/6.0 * (uy + uz) * rho ;
						f(NB) = f(ST) + N_z + 1.0/6.0 * (uy - uz) * rho ;
					}
					if (FM::isIncompressible)
					{
						uy = rho - (f(O) + f(E) + f(ET) + f(EB) + f(WB) + f(WT) + f(W) + 
								f(T) + f(B) + 2.0 * (f(SE) + f(SB) + f(ST) + f(S) + f(SW))) ;

						N_x = 0.5 * (f(E) + f(ET) + f(EB) - f(WB) - f(WT) - f(W)) - 1.0/3.0 * ux ;
						N_z = 0.5 * (f(ET) - f(EB) - f(WB) + f(WT) + f(T) - f(B)) - 1.0/3.0 * uz ;

						f(N) = f(S) + 1.0/3.0 * uy ;
						f(NE) = f(SW) - N_x + 1.0/6.0 * (ux + uy) ;
						f(NW) = f(SE) + N_x + 1.0/6.0 * ( - ux + uy) ;
						f(NT) = f(SB) - N_z + 1.0/6.0 * (uy + uz) ;
						f(NB) = f(ST) + N_z + 1.0/6.0 * (uy - uz) ;
					}
				break  ;

				case PlacementModifier::EAST  : 
					
					if (FM::isCompressible)
					{
						ux =  - 1.0 + (f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
								f(T) + f(B) + 2.0 * (f(E) + f(SE) + f(ET) + f(EB) + f(NE)))/rho ;

						N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) - 
									1.0/3.0 * uy * rho ;
						N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) - 
									1.0/3.0 * uz * rho ;

						f(W) = f(E) - 1.0/3.0 * ux * rho ;
						f(WB) = f(ET) + N_z - 1.0/6.0 * (ux + uz) * rho ;
						f(WT) = f(EB) - N_z + 1.0/6.0 * ( - ux + uz) * rho ;
						f(SW) = f(NE) + N_y - 1.0/6.0 * (ux + uy) * rho ;
						f(NW) = f(SE) - N_y + 1.0/6.0 * ( - ux + uy) * rho ;
					}
					if (FM::isIncompressible)
					{
						ux =  - rho + (f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
								f(T) + f(B) + 2.0 * (f(E) + f(SE) + f(ET) + f(EB) + f(NE))) ;

						N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) - 1.0/3.0 * uy ;
						N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) - 1.0/3.0 * uz ;

						f(W) = f(E) - 1.0/3.0 * ux ;
						f(WB) = f(ET) + N_z - 1.0/6.0 * (ux + uz) ;
						f(WT) = f(EB) - N_z + 1.0/6.0 * ( - ux + uz) ;
						f(SW) = f(NE) + N_y - 1.0/6.0 * (ux + uy) ;
						f(NW) = f(SE) - N_y + 1.0/6.0 * ( - ux + uy) ;
					}
				break  ;
				
				case PlacementModifier::WEST  : 

					if (FM::isCompressible)
					{
						ux = 1.0 - (f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
								f(T) + f(B) + 2.0 * (f(WB) + f(WT) + f(W) + f(NW) + f(SW)))/rho ;

						N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) - 
									1.0/3.0 * uy * rho ;
						N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) - 
									1.0/3.0 * uz * rho ;

						f(E) = f(W) + 1.0/3.0 * ux * rho ;
						f(ET) = f(WB) - N_z + 1.0/6.0 * (ux + uz) * rho ;
						f(EB) = f(WT) + N_z + 1.0/6.0 * (ux - uz) * rho ;
						f(NE) = f(SW) - N_y + 1.0/6.0 * (ux + uy) * rho ;
						f(SE) = f(NW) + N_y + 1.0/6.0 * (ux - uy) * rho ;
					}
					if (FM::isIncompressible)
					{
						ux = rho - (f(O) + f(NT) + f(NB) + f(SB) + f(ST) + f(N) + f(S) + 
								f(T) + f(B) + 2.0 * (f(WB) + f(WT) + f(W) + f(NW) + f(SW))) ;

						N_y = 0.5 * (f(NT) + f(NB) - f(SB) - f(ST) + f(N) - f(S)) - 1.0/3.0 * uy ;
						N_z = 0.5 * (f(NT) - f(NB) - f(SB) + f(ST) + f(T) - f(B)) - 1.0/3.0 * uz ;

						f(E) = f(W) + 1.0/3.0 * ux ;
						f(ET) = f(WB) - N_z + 1.0/6.0 * (ux + uz) ;
						f(EB) = f(WT) + N_z + 1.0/6.0 * (ux - uz) ;
						f(NE) = f(SW) - N_y + 1.0/6.0 * (ux + uy) ;
						f(SE) = f(NW) + N_y + 1.0/6.0 * (ux - uy) ;
					}
				break  ;

				case PlacementModifier::BOTTOM: 

					if (FM::isCompressible)
					{
						uz = 1.0 - (f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
								f(NW) + f(SW) + 2.0 * (f(EB) + f(WB) + f(NB) + f(SB) + f(B)))/rho ;

						N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) - 
									1.0/3.0 * ux * rho ;
						N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) - 
									1.0/3.0 * uy * rho ;

						f(T) = f(B) + 1.0/3.0 * uz * rho ;
						f(ET) = f(WB) - N_x + 1.0/6.0 * (ux + uz) * rho ;
						f(ST) = f(NB) + N_y + 1.0/6.0 * ( - uy + uz) * rho ;
						f(NT) = f(SB) - N_y + 1.0/6.0 * (uy + uz) * rho ;
						f(WT) = f(EB) + N_x + 1.0/6.0 * ( - ux + uz) * rho ;
					}
					if (FM::isIncompressible)
					{
						uz = rho - (f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
								f(NW) + f(SW) + 2.0 * (f(EB) + f(WB) + f(NB) + f(SB) + f(B))) ;

						N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) - 1.0/3.0 * ux ;
						N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) - 1.0/3.0 * uy ;

						f(T) = f(B) + 1.0/3.0 * uz ;
						f(ET) = f(WB) - N_x + 1.0/6.0 * (ux + uz) ;
						f(ST) = f(NB) + N_y + 1.0/6.0 * ( - uy + uz) ;
						f(NT) = f(SB) - N_y + 1.0/6.0 * (uy + uz) ;
						f(WT) = f(EB) + N_x + 1.0/6.0 * ( - ux + uz) ;
					}
				break  ;

				case PlacementModifier::TOP: 
					
					if (FM::isCompressible)
					{
						uz =  - 1.0 + (f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
								f(NW) + f(SW) + 2.0 * (f(ET) + f(WT) + f(NT) + f(ST) + f(T)))/rho ;

						N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) - 
									1.0/3.0 * ux * rho ;
						N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) - 
									1.0/3.0 * uy * rho ;

						f(B) = f(T) - 1.0/3.0 * uz * rho ;
						f(WB) = f(ET) + N_x - 1.0/6.0 * (ux + uz) * rho ;
						f(NB) = f(ST) - N_y + 1.0/6.0 * (uy - uz) * rho ;
						f(SB) = f(NT) + N_y - 1.0/6.0 * (uy + uz) * rho ;
						f(EB) = f(WT) - N_x + 1.0/6.0 * (ux - uz) * rho ;
					}
					if (FM::isIncompressible)
					{
						uz =  - rho + (f(O) + f(E) + f(SE) + f(N) + f(W) + f(S) + f(NE) + 
								f(NW) + f(SW) + 2.0 * (f(ET) + f(WT) + f(NT) + f(ST) + f(T))) ;

						N_x = 0.5 * (f(E) + f(SE) - f(W) + f(NE) - f(NW) - f(SW)) - 1.0/3.0 * ux ;
						N_y = 0.5 * ( - f(SE) + f(N) - f(S) + f(NE) + f(NW) - f(SW)) - 1.0/3.0 * uy ;

						f(B) = f(T) - 1.0/3.0 * uz ;
						f(WB) = f(ET) + N_x - 1.0/6.0 * (ux + uz) ;
						f(NB) = f(ST) - N_y + 1.0/6.0 * (uy - uz) ;
						f(SB) = f(NT) + N_y - 1.0/6.0 * (uy + uz) ;
						f(EB) = f(WT) - N_x + 1.0/6.0 * (ux - uz) ;
					}
				break  ;

				case PlacementModifier::EXTERNAL_EDGE                            : 
				case PlacementModifier::EXTERNAL_EDGE_PRESSURE_TANGENTIAL        : 
				case PlacementModifier::INTERNAL_EDGE                            : 
				case PlacementModifier::EXTERNAL_CORNER                          : 
				case PlacementModifier::EXTERNAL_CORNER_PRESSURE_TANGENTIAL      : 
				case PlacementModifier::CORNER_ON_EDGE_AND_PERPENDICULAR_PLANE   : 

					THROW ("Not implemented") ;

				break  ;
				
				default:
					THROW ("Undefined placementModifier for PRESSURE node") ;
			}
		break  ;

		default:
			THROW ("Unexpected node type") ;
	}


	#undef f
	#undef fPost
	#undef ux
	#undef uy
	#undef uz
	#undef uxB
	#undef uyB
	#undef uzB
	#undef rho
	#undef rhoB

}



#undef NODE_CALCULATOR
#undef TEMPLATE_NODE_CALCULATOR
#undef NODE_CALCULATOR_BASE
#undef TEMPLATE_NODE_CALCULATOR_BASE



}



#endif
