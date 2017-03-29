cmake_minimum_required(VERSION 3.6)
project(gnns)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(gnns ${SOURCE_FILES})

include_directories(algorithms util io .)
