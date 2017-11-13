cmake_minimum_required(VERSION 2.8)

IF(NOT VTK_BINARY_DIR)
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})
ENDIF(NOT VTK_BINARY_DIR)

FILE(WRITE  "tmp/vtk_settings" "VTK_INCLUDES = -I")
FILE(APPEND "tmp/vtk_settings" "/usr/local/include/vtk-5.10")
#FILE(APPEND "tmp/vtk_settings" ${VTK_INCLUDE_DIRS})
FILE(APPEND "tmp/vtk_settings" "\n")

FILE(APPEND "tmp/vtk_settings" "VTK_LIBRARY_PATH = -L")
FILE(APPEND "tmp/vtk_settings" "/usr/lib")
#FILE(APPEND "tmp/vtk_settings" ${VTK_LIBRARY_DIRS})
FILE(APPEND "tmp/vtk_settings" "\n")
