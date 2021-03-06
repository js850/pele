cmake_minimum_required(VERSION 2.8)

ADD_DEFINITIONS(-std=c++0x)

#cmake_policy(SET CMP0015 NEW)

# Add and compile the gtest library
set(gtest_src ../gtest)
set(gtest_binary_dir googletest)
set(gtest_include ${gtest_src}/include/)
include_directories(${gtest_include})
message("gtest include directory ${gtest_include}")
add_subdirectory(../gtest ${gtest_binary_dir})
set(gtest_libs gtest gtest_main)
message("gtest_libs ${gtest_libs}")


# set the pele include directory
set(pele_include ../../source)
include_directories(${pele_include})
message("pele include directory ${pele_include}")

# build the pele library
FILE(GLOB pele_sources ${pele_include}/*.cpp)
add_library(pele_lib ${pele_sources})

# get all the source files
FILE(GLOB sources *.cpp)

add_executable(test_main ${sources})
target_link_libraries(test_main pele_lib gtest gtest_main pthread)
