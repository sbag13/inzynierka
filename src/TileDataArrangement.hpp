#ifndef TILE_DATA_ARRANGEMENT_HPP
#define TILE_DATA_ARRANGEMENT_HPP



namespace microflow
{



enum class TileDataArrangement
{
	XYZ , 	// Default, all data in row-order.

	OPT_1		// Separate arrangement for all tile values, used to minimize
					// uncoalesced memory transfers while reading node values from
					// neighbor tiles.
} ;



}



#endif
