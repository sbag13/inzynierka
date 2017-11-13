#!/usr/bin/ruby

require 'rbconfig'
require 'mkmf.rb'



def includePaths
	"-I" + $hdrdir + " -I" + RbConfig::expand($arch_hdrdir)
end



def libraries
	$LIBS + " " + RbConfig::expand($LIBRUBYARG)
end



def libraryPaths
	l = $DEFLIBPATH - ['.']
	l *= " "
	"-L" + RbConfig::expand(l)
end


if "INCLUDES" == ARGV[0]
	puts includePaths
end
if "LIBRARIES" == ARGV[0]
	puts	libraries
end
if "LIBRARY_PATH" == ARGV[0]
	puts libraryPaths
end
