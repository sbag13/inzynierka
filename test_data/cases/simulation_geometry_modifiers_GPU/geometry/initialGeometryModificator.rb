puts "Running script from " + GeometryDirectory


#setNodes( coordinates(1,1,1), :baseType => fluid)

#channel1 = filledBox = box(coordinates(0,25,10), coordinates(39,35,30))

#
#	Nodes set in this stage are passed to automated node classificator, thus some
# node types may be changed based on neighbors.
#
box3x3x3 = box(coordinates(29,29,29), coordinates(31,31,31))

setNodes( box3x3x3                         ,
 					:baseType           => fluid     ,
					:placementModifier  => none      ,
					:rhoPhysical        => 0.9			 ,
					:rhoBoundaryPhysical=> 0.8       ,
					:uPhysical          => [2,2,2]   ,
					:uBoundaryPhysical  => [3,3,3]					
				)

