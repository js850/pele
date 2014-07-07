cmake_minimum_required(VERSION 2.8)

enable_language(CXX)
ADD_DEFINITIONS(-std=c++0x)
ADD_DEFINITIONS(-Wall)

#cmake_policy(SET CMP0015 NEW)

# set the pele include directory
set(pele_include ../../source)
include_directories(${pele_include})
message("pele include directory ${pele_include}")

# build the pele library
FILE(GLOB pele_sources ${pele_include}/*.cpp)
add_library(pele_lib ${pele_sources})

add_executable(lj_looping lj_looping.cpp)
target_link_libraries(lj_looping pele_lib)
