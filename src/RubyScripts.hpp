// http://stackoverflow.com/questions/1246301/c-c-can-you-include-a-file-into-a-string-literal
//TODO: Use xxd.
#define STRINGIFY(x) #x

const char * read_config_rb =
#include "read_config.rb"
;
