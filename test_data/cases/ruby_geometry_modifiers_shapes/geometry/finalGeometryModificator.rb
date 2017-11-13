puts "Running script from " + GeometryDirectory

=begin
	
	General idea of node modification.

	Method setNodes(...) requires first argument with node coordinates.
	Node coordinates may be prepared by hand or by predefined methods.
	The simplest version - single node coordinate - may be created using 
	coordinates(x,y,z) method.
	There are also other methods (e.g. box) which create simple shapes.

	Other arguments of setNodes(...) method define what shoud be modified and how.

	box(...) method creates box with two defined corners. The created box may
					 be filled or empty depending on :filled parameter.
					 Example: box( coordinates(10,10,10), coordinates(20,20,20) )


	ball( origin, radius) creates filled sphere.
					 Example: ball( coordinates(20,20,20), 10 )

=end

size = getSize()

print "\nGeometry size: "
print "width = ", size.width, ", height = ", size.height, ", depth = ", size.depth
print "\n\n"


setNodes( coordinates(1,1,1), :baseType => solid)


filledBox = box(coordinates(3,3,3), coordinates(4,4,4), :filled => true)
puts "filledBox: " + filledBox.to_s

setNodes( filledBox, :baseType => velocity )



hollowBox = box(coordinates(6,6,6), coordinates(9,9,9), :filled => false)
puts "hollowBox: " + hollowBox.to_s

setNodes( hollowBox, :baseType => velocity_0 )


#
#	Simple example of method returning shape defined as a set of nodes with 
# coordinates.
#
def myShape
	result = Array.new

	result << coordinates(0,0,0)

	return result
end

setNodes( myShape(), :baseType => pressure )

puts "myShape: " + myShape().to_s


ball_1 = ball( coordinates(20,20,20), 8 )

setNodes( ball_1, :baseType => pressure )

puts "ball_1: " + ball_1.to_s

