#
# Application Build File
#

cmake_minimum_required( VERSION 3.8 FATAL_ERROR )

# Set Project Name
set( PNAME cfdbox )

project( ${PNAME}_project LANGUAGES CXX CUDA )

include (TestBigEndian)
TEST_BIG_ENDIAN(U_BIG_ENDIAN)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release"
        CACHE STRING "Choose the type of build, options are: Debug Release
        RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Add  Dependencies Libraries
#find_package(u REQUIRED)
#include_directories(${u_INCLUDE_DIRS})
#target_link_libraries(${PNAME} u)

find_package(OpenGL REQUIRED)
include_directories(${PNAME} ${OPENGL_INCLUDE_DIR})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL CACHE BOOL "" FORCE)
add_subdirectory(glfw)

add_definitions(-DCMAKE_SECURE_NO_DEPRECATE -DCMAKE_SECURE_NO_WARNINGS)
include_directories(${PNAME} ${PROJECT_SOURCE_DIR}/nuklear)

# Find all C++ source files
file(GLOB_RECURSE HDRS
    ${PROJECT_SOURCE_DIR}/src/*.h
    ${PROJECT_SOURCE_DIR}/src/*.hh
    ${PROJECT_SOURCE_DIR}/src/*.hpp )
file(GLOB_RECURSE SRCS
    ${PROJECT_SOURCE_DIR}/src/*.c
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Add Include Directories
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}/src")

# Create Executable 
add_executable( ${PNAME} ${SRCS} )
target_link_libraries(${PNAME} glfw)
target_link_libraries(${PNAME} ${OPENGL_LIBRARIES})



#Debugging print all CMAKE Variables
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

configure_file(proj-config.cmake.in "${PROJECT_BINARY_DIR}/${PNAME}-config.cmake" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/src/config.hh.in" "${PROJECT_BINARY_DIR}/src/config.hh")

install(FILES "${PROJECT_BINARY_DIR}/${PNAME}-config.cmake" DESTINATION lib/${PNAME})
install( TARGETS ${PNAME}
    EXPORT ${PNAME} 
    DESTINATION bin )
install(EXPORT ${PNAME} DESTINATION bin)

