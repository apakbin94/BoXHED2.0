cmake_minimum_required(VERSION 3.18.2)
PROJECT(boxhed_preprocessor VERSION 1.0)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")

add_library(_preprocessor MODULE src/preprocessor.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
        target_link_libraries(_preprocessor OpenMP::OpenMP_CXX)
endif()
