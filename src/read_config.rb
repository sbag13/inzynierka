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

def undef_const(c)
	if Object.const_defined?(c) then Object.send(:remove_const, c) end ;
end ;

undef_const( :D3Q19             ) ;
undef_const( :MRT               ) ;
undef_const( :BGK               ) ;
undef_const( :BINARY            ) ;
undef_const( :ASCII             ) ;
undef_const( :DOUBLE            ) ;
undef_const( :FLOAT             ) ;
undef_const( :Re                ) ;
undef_const( :CPU               ) ;
undef_const( :GPU               ) ;

D3Q19 = "D3Q19" ;
single = "single" ;
double = "double" ;
CPU = "CPU" ;
GPU = "GPU" ;

incompressible = "incompressible" ;
quasi_compressible = "quasi_compressible" ;

MRT = "MRT" ;
BGK = "BGK" ;

BINARY = "BINARY" ;
ASCII  = "ASCII"  ;
DOUBLE = "double" ;
FLOAT  = "float"  ;

mean = "mean" ;
nan  = "nan"  ;

yes = true  ;
no  = false ;



$BINDING = binding() ;



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



SEPARATOR = ":_:" ;



def defineNodeTypeMethod (name)
	cmd_str = 
		"def " + name + " (placement_modifier = \"none\") return \"" + name + "\" + SEPARATOR + placement_modifier.to_s ; end ;" ;

	$BINDING.eval cmd_str.to_s ;
end ;



defineNodeTypeMethod ("velocity")      ;
defineNodeTypeMethod ("velocity_0")    ;
defineNodeTypeMethod ("pressure")      ;
defineNodeTypeMethod ("bounce_back_2") ;
defineNodeTypeMethod ("solid")         ;
defineNodeTypeMethod ("fluid")         ;



def read_conf_file( f_name ) 

	full_name = Configuration_dir + f_name ;
	f = File.new( full_name, "r") ;

	$BINDING.eval( "z_expand_depth = 1 ;" ) ;
	$BINDING.eval( "uz0_LB = 0 ;" ) ;

	line_nr = 1 ;
	f.each_line  { |line|

		$BINDING.eval( line.to_s, full_name, line_nr ) ; 

		line_nr += 1 ;		
	} ;

end ;



def export_to_global( var_name )
	
	cmd_str = "$" + var_name + " = " + var_name ;

	$BINDING.eval cmd_str.to_s ;
end ;



begin 
	
	$BINDING.eval "l_n = Ln ;\n" ;

	read_conf_file( "program_params"      ) ;
                                        
	export_to_global( "save_vtk_steps"    ) ;

	export_to_global( "vtkDefaultRhoForBB2Nodes"  ) ;

	$BINDING.eval "$number_vtk_saves = number_vtk_saves"     \n

	export_to_global( "numberOfStepsBetweenCheckpointSaves" ) ;
	export_to_global( "maxNumberOfCheckpoints" ) ;

	export_to_global( "error_print_steps" ) ;

	
	read_conf_file( "case_params" ) ;

	export_to_global( "fluid_model" ) ;
	export_to_global( "lattice" ) ;
	export_to_global( "data_type" ) ;
	export_to_global( "collision_model" ) ;
	export_to_global( "computational_engine" ) ;

	export_to_global( "z_expand_depth" ) ;

	export_to_global( "rho0_LB " ) ;
	export_to_global( "ux0_LB  " ) ;
	export_to_global( "uy0_LB  " ) ;
	export_to_global( "uz0_LB  " ) ;

	export_to_global( "nu_phys   " ) ;
	export_to_global( "rho0_phys " ) ;

	export_to_global( "l_ch_phys " ) ;
	export_to_global( "u_ch_phys " ) ;
	export_to_global( "l_ch_LB   " ) ;
	export_to_global( "err"        ) ;
	export_to_global( "tau"        ) ;

	export_to_global( "vtk_save_velocity_LB"         ) ;
	export_to_global( "vtk_save_velocity_physical"   ) ;
	export_to_global( "vtk_save_rho_LB"              ) ;
	export_to_global( "vtk_save_pressure_physical"   ) ;
	export_to_global( "vtk_save_nodes"               ) ;
	export_to_global( "vtk_save_mass_flow_fractions" ) ;


	$BINDING.eval( "$Nx = Nx " ) ;
	$BINDING.eval( "$Ny = Ny " ) ;

  export_to_global ("defaultWallNode"                   ) ;
  export_to_global ("defaultExternalCornerNode"         ) ;
  export_to_global ("defaultInternalCornerNode"         ) ;
  export_to_global ("defaultExternalEdgeNode"           ) ;
  export_to_global ("defaultInternalEdgeNode"           ) ;
  export_to_global ("defaultNotIdentifiedNode"          ) ;
  export_to_global ("defaultExternalEdgePressureNode"   ) ;
  export_to_global ("defaultExternalCornerPressureNode" ) ;
  export_to_global ("defaultEdgeToPerpendicularWallNode") ;

rescue Exception => exc ; 
	puts  exc ; 
	throw ;
end ; 
\n
)
