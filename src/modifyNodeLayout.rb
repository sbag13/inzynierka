STRINGIFY(

\n=begin \n
	Script needed mainly for loading and interpretation of configuration files. \n
	From ruby it is easy to handle exceptions and print nice message.\n
	\n
	This script is loaded as stringified macro in C++ thus some hacks are needed to
	preserve correctness after stringification. \n
	1. Stringification replaces all end of line characters with spaces thus each
	ruby line has to be ended with semicolon. \n
	2. If end of line is required then must be written explicitly as \\n. \n
	3. Commas has to be avoided unless inside brackets. \n
	4. Ruby comments can not be marked with hash since this is treated as macro. \n
	5. In commens end of line must be explicitly written before =end  because 1. \n
	\n
		These restrictions does not apply to loaded configuration files. 
	\n
\n=end \n


include Math \n


velocity      = "velocity"      ;
velocity_0    = "velocity_0"    ;
pressure      = "pressure"      ;
bounce_back_2 = "bounce_back_2" ;
solid         = "solid"         ;
fluid         = "fluid"         ;


none                                   = "none" ;
north                                  = "north" ;
south                                  = "south" ;
east                                   = "east" ;
west                                   = "west" ;
bottom                                 = "bottom" ;
top                                    = "top" ;
external_edge                          = "external_edge" ;
external_edge_pressure_tangential      = "external_edge_pressure_tangential" ;
internal_edge                          = "internal_edge" ;
external_corner                        = "external_corner" ;
external_corner_pressure_tangential    = "external_corner_pressure_tangential" ;
corner_on_edge_and_perpendicular_plane = "corner_on_edge_and_perpendicular_plane" ;

\n



class Coordinates \n
	attr_accessor :x \n
	attr_accessor :y \n
	attr_accessor :z \n

\n=begin \n
	http://www.reactive.io/tips/2008/12/21/understanding-ruby-blocks-procs-and-lambdas/
\n=end \n

	def each          \n
		yield( self )   \n
	end               \n


	def to_s           \n
		return "(x=" + @x.to_s + ",y=" + @y.to_s + ",z=" + @z.to_s + ")" \n
	end                \n


	def -(scalar)              \n
		result = dup             \n
                             \n
		result.x -= scalar       \n
		result.y -= scalar       \n
		result.z -= scalar       \n
                             \n
		return result            \n
	end                        \n


	def +(scalar)              \n
		result = dup             \n
                             \n
		result.x += scalar       \n
		result.y += scalar       \n
		result.z += scalar       \n
                             \n
		return result            \n
	end                        \n
	

end \n



def coordinates( x,y,z )

	result = Coordinates.new ;

	result.x = x ;
	result.y = y ;
	result.z = z ;

	return result
end ;



def box( corner1, corner2, params={} )

	result = Array.new ;

	defaultParams = {} ;
	defaultParams[:filled] = true ;

	params = defaultParams.merge( params ) ;

	if params[:filled] then                                     \n
                                                              
		for x in corner1.x .. corner2.x do                        \n
			for y in corner1.y .. corner2.y do                      \n
				for z in corner1.z .. corner2.z do                    \n

					result.push( coordinates(x,y,z) ) ;

				end ;                                                 \n
			end ;                                                   \n
		end ;                                                     \n

	else                                                        \n

		for x in corner1.x .. corner2.x do                        \n
			for y in corner1.y .. corner2.y do                      \n

					result.push( coordinates(x,y,corner1.z) ) ;
					result.push( coordinates(x,y,corner2.z) ) ;

			end ;                                                   \n
		end ;                                                     \n

\n=begin \n
	FIXME: Edges are filled twice.
\n=end \n

		for x in corner1.x .. corner2.x do                        \n
			for z in corner1.z .. corner2.z do                      \n

					result.push( coordinates(x,corner1.y,z) ) ;
					result.push( coordinates(x,corner2.y,z) ) ;

			end ;                                                   \n
		end ;                                                     \n
		for y in corner1.y .. corner2.y do                        \n
			for z in corner1.z .. corner2.z do                      \n

					result.push( coordinates(corner1.x,y,z) ) ;
					result.push( coordinates(corner2.x,y,z) ) ;

			end ;                                                   \n
		end ;                                                     \n

		result.uniq!

	end ;

	return result ;
end ;





\n
def ball( origin, radius )                                      \n
                                                                \n
	r = radius                                                    \n
	cube = box( origin - r , origin + r )                         \n
                                                                \n
	result = []                                                   \n
                                                                \n
	cube.each do |node|                                           \n
                                                                \n
		distance = sqrt(                                            \n
							(node.x - origin.x) ** 2 +                        \n
							(node.y - origin.y) ** 2 +                        \n
							(node.z - origin.z) ** 2                          \n
						)                                                   \n
                                                                \n
		if distance <= radius then                                  \n
			result << node                                            \n
		end			                                                    \n
                                                                \n
	end                                                           \n
                                                                \n
	return result                                                 \n
end                                                             \n


def setNodes( coordinates, arguments={} )                                              \n
                                                                                       \n
	defaultNodeData = {} ;
	defaultNodeData[:baseType           ] = nil ;
	defaultNodeData[:placementModifier  ] = nil ;
	defaultNodeData[:rhoPhysical        ] = nil ;
	defaultNodeData[:rhoBoundaryPhysical] = nil ;
	defaultNodeData[:uPhysical          ] = nil ;
	defaultNodeData[:uBoundaryPhysical  ] = nil ;

	arguments = defaultNodeData.merge( arguments ) ;

	coordinates.each do |c|                                                              \n

		baseType            = arguments[:baseType           ] ;
		placementModifier   = arguments[:placementModifier  ] ;
		rhoPhysical         = arguments[:rhoPhysical        ] ;
		rhoBoundaryPhysical = arguments[:rhoBoundaryPhysical] ;
		uPhysical           = arguments[:uPhysical          ] ;
		uBoundaryPhysical   = arguments[:uBoundaryPhysical  ] ;

		if nil != baseType then                                                            \n
			setNodeBaseType( c.x, c.y, c.z, baseType ) ;                                     \n
		end                                                                                \n

		if nil != placementModifier then                                                   \n
			setNodePlacementModifier( c.x, c.y, c.z, placementModifier ) ;                   \n
		end                                                                                \n

		if nil != rhoPhysical then                                                         \n
			setNodeRhoPhysical( c.x, c.y, c.z, rhoPhysical ) ;                               \n
		end                                                                                \n

		if nil != rhoBoundaryPhysical then                                                 \n
			setNodeRhoBoundaryPhysical( c.x, c.y, c.z, rhoBoundaryPhysical ) ;               \n
		end                                                                                \n

		if nil != uPhysical and uPhysical.kind_of?(Array) then                             \n
			if 3 != uPhysical.size then                                                      \n
				puts "Size of uPhysical = " + uPhysical.size.to_s + " differs from 3"          \n
			else                                                                             \n
				setNodeUPhysical( c.x, c.y, c.z, uPhysical ) ;                                 \n
			end                                                                              \n
		end                                                                                \n

		if nil != uBoundaryPhysical and uBoundaryPhysical.kind_of?(Array) then             \n
			if 3 != uBoundaryPhysical.size then                                              \n
				puts "Size of uBoundaryPhysical = " + uBoundaryPhysical.size.to_s +            \n
						 " differs from 3"                                                         \n
			else                                                                             \n
				setNodeUBoundaryPhysical( c.x, c.y, c.z, uBoundaryPhysical ) ;                 \n
			end                                                                              \n
		end                                                                                \n


	end ;
end ;




\n
class Node \n
	attr_accessor :baseType \n
	attr_accessor :placementModifier \n

	def to_s()
		@baseType + " " + @placementModifier + " " ;
	end ;
end \n




\n
class Size \n
	attr_accessor :width \n
	attr_accessor :height \n
	attr_accessor :depth \n

	def to_s \n
		"width=" + @width.to_s + ", height=" + @height.to_s + ", depth=" + @depth.to_s ;
	end\n
end \n




\n
)
