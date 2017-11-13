puts "Running script from " + GeometryDirectory

=begin
	
	General idea of node modification.

	Method setNodes(...) requires first argument with node coordinates.
	Node coordinates may be prepared by hand or by predefined methods.
	The simplest version - single node coordinate - may be created using 
	coordinates(x,y,z) method.
	There are also other methods (e.g. box) which create simple shapes.

	Other arguments of setNodes(...) method define what shoud be modified and how.
	The available arguments are:

 					:baseType           
					:placementModifier  
					:rhoPhysical        
					:rhoBoundaryPhysical
					:uPhysical          
					:uBoundaryPhysical  		

	box(...) method creates box with two defined corners. The created box may
					 be filled or empty depending on :filled parameter.

=end

setNodes( coordinates( 31,31,31)           ,
 					:baseType           => velocity  ,
# We can not use placement modifier "none" because this modification is done
# after automated node classification.
					:placementModifier  => top       , 
					:rhoPhysical        => 0.5			 ,
					:rhoBoundaryPhysical=> 0.4       ,
					:uPhysical          => [4,4,4]   ,
					:uBoundaryPhysical  => [5,5,5]					
				)


