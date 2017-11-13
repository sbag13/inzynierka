#include "gtest/gtest.h"
#include "kernelTileGatherProcessBoundaryCollide.tcc"
#include "LatticeArrangementD3Q19.hpp"
#include "Storage.hpp"
#include "Tile.hpp"

#include "NodeLayoutTest.hpp"
#include "TiledLatticeTest.hpp"



using namespace microflow ;
using namespace std ;



static __device__ unsigned neighborTileIndexMapCopyDevice [3][3][3] ;



/*
	FIXME: Kernel is only partially adapted to full use of template parameters.
				 Currently only DataArrangement can be changed, the rest of template
				 parameters require carefull examination of the code.
*/
template <class LatticeArrangement, class DataType, unsigned Edge,
					TileDataArrangement DataArrangement>
__global__ void
kernelLoadCopyOfTileMapTest
(
	size_t widthInNodes,
	size_t heightInNodes,
	size_t depthInNodes,

	unsigned int * __restrict__ tileMap,

	size_t nOfNonEmptyTiles,
	size_t * __restrict__ nonEmptyTilesX0,
	size_t * __restrict__ nonEmptyTilesY0,
	size_t * __restrict__ nonEmptyTilesZ0
)
{
	__shared__ unsigned int neighborTileIndexMapCopy [3][3][3] ;

	typedef Tile <LatticeArrangement, DataType, Edge, StorageInKernel, DataArrangement> TileType ;
	const unsigned tileIndex = blockIdx.x ;

	loadCopyOfTileMap <TileType>
	(
		neighborTileIndexMapCopy,

		widthInNodes, heightInNodes, depthInNodes,
		tileMap,
		nOfNonEmptyTiles,
		nonEmptyTilesX0, nonEmptyTilesY0, nonEmptyTilesZ0,
		tileIndex
	) ;


	// FIXME: Do tests using warp level programming (without __syncthreads()).	
	__syncthreads() ; 


	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
		unsigned bidx = blockIdx.x ;
		printf ("Tile %d: x0=%llu, y0=%llu, z0=%llu\n", bidx,
						nonEmptyTilesX0[bidx], nonEmptyTilesY0[bidx], nonEmptyTilesZ0[bidx]) ;
	}

	if (blockIdx.x == 13 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
		for (unsigned i=0 ; i < 27 ; i++)
		{
			printf("%d\t", tileMap[i]) ;
		}
		printf ("\n") ;

		for (unsigned z=0 ; z < 3 ; z++)
			for (unsigned y=0 ; y < 3 ; y++)
				for (unsigned x=0 ; x < 3 ; x++)
				{
					printf ("%d  ",neighborTileIndexMapCopy[z][y][x]) ;
					neighborTileIndexMapCopyDevice[z][y][x] = neighborTileIndexMapCopy[z][y][x] ;
				}

		printf ("\n") ;
	}
}



template <class LatticeArrangement, class DataType, unsigned Edge,
					TileDataArrangement DataArrangement>
void testLoadCopyOfTileMap()					
{
	NodeLayout nodeLayout = createFluidNodeLayout (4*3, 4*3, 4*3) ; // 3x3x3 tiles

	ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
	TileLayout<StorageOnCPU> tileLayoutCPU (nodeLayout)  ;

	TileLayout<StorageOnGPU> tileLayoutGPU (tileLayoutCPU) ;

	typedef TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement> TLatticeCPU ;
	typedef TiledLattice <LatticeArrangement,DataType,StorageOnGPU,DataArrangement> TLatticeGPU ;

	TLatticeCPU tiledLatticeCPU (tileLayoutCPU, expandedNodeLayout, Settings()) ;
	TLatticeGPU tiledLatticeGPU (tiledLatticeCPU, tileLayoutGPU) ;

	dim3 numBlocks (27,1,1) ;
	dim3 numThreads (4,4,4) ;
	Size sizeInNodes = nodeLayout.getSize() ;

	cout << "Size = " << sizeInNodes << "\n" ;

	kernelLoadCopyOfTileMapTest <LatticeArrangement,DataType,Edge,DataArrangement>
		<<<numBlocks, numThreads>>>
		(
		 sizeInNodes.getWidth(),
		 sizeInNodes.getHeight(),
		 sizeInNodes.getDepth(),

		 tiledLatticeGPU.getTileLayout().getTileMapPointer(),

		 tiledLatticeGPU.getTileLayout().getNoNonEmptyTiles(),

		 tiledLatticeGPU.getTileLayout().getTilesX0Pointer(),
		 tiledLatticeGPU.getTileLayout().getTilesY0Pointer(),
		 tiledLatticeGPU.getTileLayout().getTilesZ0Pointer()
		) ;

	CUDA_CHECK( cudaPeekAtLastError() );
	CUDA_CHECK( cudaDeviceSynchronize() );  	

	unsigned neighborTileIndexMapCopyHost [3][3][3] ;
	CUDA_CHECK(
			cudaMemcpyFromSymbol( &neighborTileIndexMapCopyHost, 
														 neighborTileIndexMapCopyDevice, 
														sizeof(neighborTileIndexMapCopyHost), 
														0, cudaMemcpyDeviceToHost)
			) ;

	auto tileMap = tileLayoutCPU.getTileMap() ;

	cout << "CPU tile map:\n" ;
	for (unsigned z=0 ; z < 3 ; z++)
	{
		cout << "z=" << z << "\n" ;
		for (unsigned y=0 ; y < 3 ; y++)
		{
			cout << "\ty=" << y << ":\t\t" ;
			for (unsigned x=0 ; x < 3 ; x++)
			{
				cout << tileMap[ Coordinates(x,y,z) ] << "\t" ;
			}
			cout << "\n" ;
		}
	}

	cout << "GPU tile map:\n" ;
	for (unsigned z=0 ; z < 3 ; z++)
	{
		cout << "z=" << z << "\n" ;
		for (unsigned y=0 ; y < 3 ; y++)
		{
			cout << "\ty=" << y << ":\t\t" ;
			for (unsigned x=0 ; x < 3 ; x++)
			{
				cout << neighborTileIndexMapCopyHost[z][y][x] << "\t" ;
			}
			cout << "\n" ;
		}
	}

	for (unsigned z=0 ; z < 3 ; z++)
		for (unsigned y=0 ; y < 3 ; y++)
			for (unsigned x=0 ; x < 3 ; x++)
			{
				EXPECT_EQ (tileMap[ Coordinates(x,y,z) ], neighborTileIndexMapCopyHost[z][y][x] ) 
					<< " error for x=" << x << ", y= " << y <<", z= " << z << "\n" ;
			}
}



TEST (kernelTileGatherProcessBoundaryCollideTest_XYZ, loadCopyOfTileMap)
{
	testLoadCopyOfTileMap <D3Q19,double,4u, TileDataArrangement::XYZ> () ;
}



TEST (kernelTileGatherProcessBoundaryCollideTest_OPT_1, loadCopyOfTileMap)
{
	testLoadCopyOfTileMap <D3Q19,double,4u, TileDataArrangement::OPT_1> () ;
}



static __device__
NodeType::PackedDataType nodeTypesCopyDevice [6][6][6] ;



/*
	FIXME: Kernel is only partially adapted to full use of template parameters.
				 Currently only DataArrangement can be changed, the rest of template
				 parameters require carefull examination of the code.
*/
template <class LatticeArrangement, class DataType, unsigned Edge,
					TileDataArrangement DataArrangement>
__global__ void
kernelLoadCopyOfNodeTypesTest
(
	size_t widthInNodes,
	size_t heightInNodes,
	size_t depthInNodes,

	unsigned int * __restrict__ tileMap,

	size_t nOfNonEmptyTiles,
	size_t * __restrict__ nonEmptyTilesX0,
	size_t * __restrict__ nonEmptyTilesY0,
	size_t * __restrict__ nonEmptyTilesZ0,

	NodeType          * __restrict__ tiledNodeTypes
)
{
	typedef Tile <LatticeArrangement,DataType,Edge,StorageInKernel,DataArrangement> TileType ;

	const unsigned tileIndex = blockIdx.x ;

	__shared__ unsigned int neighborTileIndexMapCopy [3][3][3] ;
	__shared__ NodeType::PackedDataType nodeTypesCopy [Edge+2][Edge+2][Edge+2] ;

	loadCopyOfTileMap <TileType>
	(
		neighborTileIndexMapCopy,

		widthInNodes, heightInNodes, depthInNodes,
		tileMap,
		nOfNonEmptyTiles,
		nonEmptyTilesX0, nonEmptyTilesY0, nonEmptyTilesZ0,
		tileIndex
	) ;


	loadCopyOfNodeTypes <TileType>
	(
		nodeTypesCopy,
		neighborTileIndexMapCopy,
		tiledNodeTypes,
		tileIndex		
	) ;


	// FIXME: Do tests using warp level programming (without __syncthreads()).	
	__syncthreads() ; 


	if (blockIdx.x == 13 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
		for (unsigned i=0 ; i < 27 ; i++)
		{
			printf("%d\t", tileMap[i]) ;
		}
		printf ("\n") ;

		for (unsigned z=0 ; z < 3 ; z++)
			for (unsigned y=0 ; y < 3 ; y++)
				for (unsigned x=0 ; x < 3 ; x++)
				{
					neighborTileIndexMapCopyDevice[z][y][x] = neighborTileIndexMapCopy[z][y][x] ;
				}

		for (unsigned z=0 ; z < 6 ; z++)
			for (unsigned y=0 ; y < 6 ; y++)
				for (unsigned x=0 ; x < 6 ; x++)
				{
					nodeTypesCopyDevice [z][y][x] = nodeTypesCopy [z][y][x] ;
				}

	}
}
	


void compareNodes (Coordinates tileCorner, int x, int y, int z,
									 NodeLayout & nodeLayout,
									 NodeType::PackedDataType (&nodeTypesCopyHost) [6][6][6]
									 )
{
	Coordinates nodeCoordinates = tileCorner + Coordinates(x,y,z) ;
	auto nodeCPU = nodeLayout.safeGetNodeType(nodeCoordinates) ;
	NodeType nodeGPU ;
	nodeGPU.packedData() = nodeTypesCopyHost[z+1][y+1][x+1] ;

	ASSERT_EQ (nodeCPU, nodeGPU) 
		<< " error for x =" << x << ", y = " << y <<", z = " << z 
		<< ", nodeCPU = " << nodeCPU.packedData() 
		<< ", nodeGPU = " << nodeGPU.packedData()
		<< "\n" ;	
}



template <class LatticeArrangement, class DataType, unsigned Edge,
					TileDataArrangement DataArrangement>
void testLoadCopyOfNodeTypes()
{
		NodeLayout nodeLayout = createFluidNodeLayout (Edge*3, Edge*3, Edge*3) ; // 3x3x3 tiles

		Size nodeLayoutSize = nodeLayout.getSize() ;
		unsigned baseType = 0 ;
		for (unsigned z=0 ; z < nodeLayoutSize.getDepth() ; z++)
			for (unsigned y=0 ; y < nodeLayoutSize.getHeight() ; y++)
				for (unsigned x=0 ; x < nodeLayoutSize.getWidth() ; x++)
				{
					nodeLayout.setNodeType (x,y,z, NodeType (static_cast<NodeBaseType>(baseType)) ) ;
					baseType += 1 ;
					baseType %= static_cast<unsigned>(NodeBaseType::PRESSURE) ;
					baseType += 1 ;
				}

		ExpandedNodeLayout expandedNodeLayout (nodeLayout) ;
		TileLayout<StorageOnCPU> tileLayoutCPU (nodeLayout)  ;

		TileLayout<StorageOnGPU> tileLayoutGPU (tileLayoutCPU) ;

		typedef TiledLattice <LatticeArrangement,DataType,StorageOnCPU,DataArrangement> TLatticeCPU ;
		typedef TiledLattice <LatticeArrangement,DataType,StorageOnGPU,DataArrangement> TLatticeGPU ;

		TLatticeCPU tiledLatticeCPU (tileLayoutCPU, expandedNodeLayout, Settings()) ;
		TLatticeGPU tiledLatticeGPU (tiledLatticeCPU, tileLayoutGPU) ;

		dim3 numBlocks (27,1,1) ;
		dim3 numThreads (Edge,Edge,Edge) ;
		Size sizeInNodes = nodeLayout.getSize() ;

		cout << "Size = " << sizeInNodes << "\n" ;

		kernelLoadCopyOfNodeTypesTest <LatticeArrangement,DataType,Edge,DataArrangement>
			<<<numBlocks, numThreads>>>
			(
			 sizeInNodes.getWidth(),
			 sizeInNodes.getHeight(),
			 sizeInNodes.getDepth(),

			 tiledLatticeGPU.getTileLayout().getTileMapPointer(),

			 tiledLatticeGPU.getTileLayout().getNoNonEmptyTiles(),

			 tiledLatticeGPU.getTileLayout().getTilesX0Pointer(),
			 tiledLatticeGPU.getTileLayout().getTilesY0Pointer(),
			 tiledLatticeGPU.getTileLayout().getTilesZ0Pointer(),

			 tiledLatticeGPU.getNodeTypesPointer()
			) ;

		CUDA_CHECK( cudaPeekAtLastError() );
		CUDA_CHECK( cudaDeviceSynchronize() );  	

		unsigned neighborTileIndexMapCopyHost [3][3][3] ;
		CUDA_CHECK(
				cudaMemcpyFromSymbol( &neighborTileIndexMapCopyHost, 
															 neighborTileIndexMapCopyDevice, 
															sizeof(neighborTileIndexMapCopyHost), 
															0, cudaMemcpyDeviceToHost)
				) ;

		NodeType::PackedDataType nodeTypesCopyHost [6][6][6] ;
		CUDA_CHECK(
				cudaMemcpyFromSymbol( &nodeTypesCopyHost, 
															 nodeTypesCopyDevice, 
															sizeof(nodeTypesCopyHost), 
															0, cudaMemcpyDeviceToHost)
				) ;

		auto tileMap = tileLayoutCPU.getTileMap() ;

		cout << "CPU tile map:\n" ;
		for (unsigned z=0 ; z < 3 ; z++)
		{
			cout << "z=" << z << "\n" ;
			for (unsigned y=0 ; y < 3 ; y++)
			{
				cout << "\ty=" << y << ":\t\t" ;
				for (unsigned x=0 ; x < 3 ; x++)
				{
					cout << tileMap[ Coordinates(x,y,z) ] << "\t" ;
				}
				cout << "\n" ;
			}
		}

		cout << "GPU tile map:\n" ;
		for (unsigned z=0 ; z < 3 ; z++)
		{
			cout << "z=" << z << "\n" ;
			for (unsigned y=0 ; y < 3 ; y++)
			{
				cout << "\ty=" << y << ":\t\t" ;
				for (unsigned x=0 ; x < 3 ; x++)
				{
					cout << neighborTileIndexMapCopyHost[z][y][x] << "\t" ;
				}
				cout << "\n" ;
			}
		}

		for (unsigned z=0 ; z < 3 ; z++)
			for (unsigned y=0 ; y < 3 ; y++)
				for (unsigned x=0 ; x < 3 ; x++)
				{
					EXPECT_EQ (tileMap[ Coordinates(x,y,z) ], neighborTileIndexMapCopyHost[z][y][x] ) 
						<< " error for x=" << x << ", y= " << y <<", z= " << z << "\n" ;
				}


		auto tile13 = tileLayoutCPU.getTile (13) ;
		Coordinates tileCorner = tile13.getCornerPosition() ;
		cout << "CPU node types:\n" ;
		for (int z=5 ; z >= 0 ; z--)
		{
			cout << "z=" << z << "\n" ;
			for (int y=5 ; y >= 0 ; y--)
			{
				cout << "\ty=" << y << ":\t\t" ;
				for (int x=5 ; x >= 0 ; x--)
				{
					Coordinates nodeCoordinates = tileCorner - Coordinates(1,1,1) + Coordinates(x,y,z) ;
					auto node = nodeLayout.safeGetNodeType(nodeCoordinates) ;
					cout << node.packedData() << "\t" ;
				}
				cout << "\n" ;
			}
		}

		cout << "GPU node types:\n" ;
		for (int z=5 ; z >= 0 ; z--)
		{
			cout << "z=" << z << "\n" ;
			for (int y=5 ; y >= 0 ; y--)
			{
				cout << "\ty=" << y << ":\t\t" ;
				for (int x=5 ; x >= 0 ; x--)
				{
					cout << nodeTypesCopyHost[z][y][x] << "\t" ;
				}
				cout << "\n" ;
			}
		}

		
		cout << "Comparing node types inside tile..." << flush ;
		for (unsigned z=0 ; z < 4 ; z++)
			for (unsigned y=0 ; y < 4 ; y++)
				for (unsigned x=0 ; x < 4 ; x++)
				{
					compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
				}
		cout << " OK\n" ;

		cout << "Comparing planes on east and west..." << flush ;
		for (unsigned z=0 ; z < 4 ; z++)
			for (unsigned y=0 ; y < 4 ; y++)
			{
				{
					int x = -1 ;
					compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
				}
				{
					unsigned x = 4 ;
					compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
				}
			}
		cout << " OK\n" ;

		cout << "Comparing planes on south and north..." << flush ;
		for (unsigned z=0 ; z < 4 ; z++)
			for (unsigned x=0 ; x < 4 ; x++)
			{
				{
					int y = -1 ;
					compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
				}
				{
					unsigned y = 4 ;
					compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
				}
			}
		cout << " OK\n" ;

		cout << "Comparing planes on top and bottom..." << flush ;
		for (unsigned y=0 ; y < 4 ; y++)
			for (unsigned x=0 ; x < 4 ; x++)
			{
				{
					int z = -1 ;
					compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
				}
				{
					unsigned z = 4 ;
					compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
				}
			}
		cout << " OK\n" ;

		cout << "Comparing 4 parallel edges in center of the tile (from west to east)..." << flush ;
		{
			int y = -1, z = -1 ;
			for (unsigned x=0 ; x < 4 ; x++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			y = -1 ; z = 4 ;
			for (unsigned x=0 ; x < 4 ; x++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			y = 4 ; z = -1 ;
			for (unsigned x=0 ; x < 4 ; x++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			y = 4 ; z = 4 ;
			for (unsigned x=0 ; x < 4 ; x++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
		}
		cout << " OK.\n" ;

		cout << "Comparing 4 parallel vertical edges (from bottom to top)..." << flush ;
		{
			int x = -1, y = -1 ;
			for (unsigned z=0 ; z < 4 ; z++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			x = -1 ; y = 4 ;
			for (unsigned z=0 ; z < 4 ; z++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			x = 4 ; y = -1 ;
			for (unsigned z=0 ; z < 4 ; z++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			x = 4 ; y = 4 ;
			for (unsigned z=0 ; z < 4 ; z++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
		}
		cout << " OK.\n" ;

		cout << "Comparing 4 parallel horizontal edges (from south to east)..." << flush ;
		{
			int x = -1, z = -1 ;
			for (unsigned y=0 ; y < 4 ; y++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			x = -1 ; z = 4 ;
			for (unsigned y=0 ; y < 4 ; y++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			x = 4 ; z = -1 ;
			for (unsigned y=0 ; y < 4 ; y++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;

			x = 4 ; z = 4 ;
			for (unsigned y=0 ; y < 4 ; y++)
				compareNodes (tileCorner, x,y,z, nodeLayout, nodeTypesCopyHost) ;
		}
		cout << " OK.\n" ;

		cout << "Comparing 8 vertices..." << flush ;
		{
			compareNodes (tileCorner, -1,-1,-1, nodeLayout, nodeTypesCopyHost) ;
			compareNodes (tileCorner, -1,-1, 4, nodeLayout, nodeTypesCopyHost) ;
			compareNodes (tileCorner, -1, 4,-1, nodeLayout, nodeTypesCopyHost) ;
			compareNodes (tileCorner, -1, 4, 4, nodeLayout, nodeTypesCopyHost) ;
			compareNodes (tileCorner,  4,-1,-1, nodeLayout, nodeTypesCopyHost) ;
			compareNodes (tileCorner,  4,-1, 4, nodeLayout, nodeTypesCopyHost) ;
			compareNodes (tileCorner,  4, 4,-1, nodeLayout, nodeTypesCopyHost) ;
			compareNodes (tileCorner,  4, 4, 4, nodeLayout, nodeTypesCopyHost) ;
		}
		cout << " OK.\n" ;
}



TEST (kernelTileGatherProcessBoundaryCollideTest_XYZ, loadCopyOfNodeTypes)
{
	testLoadCopyOfNodeTypes <D3Q19,double,4u,TileDataArrangement::XYZ> () ;
}



TEST (kernelTileGatherProcessBoundaryCollideTest_OPT_1, loadCopyOfNodeTypes)
{
	testLoadCopyOfNodeTypes <D3Q19,double,4u,TileDataArrangement::OPT_1> () ;
}

