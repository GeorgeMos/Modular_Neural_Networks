cmake_minimum_required(VERSION 3.1)
 
project(ml LANGUAGES CXX)



file(GLOB_RECURSE LIBS_INCLUDE lib/include/*.h)
file(GLOB_RECURSE LIBS_SRC lib/src/*.cpp)

find_package(OpenMP)

add_executable(ml main.cpp ${LIBS_INCLUDE} ${LIBS_SRC})

if(OpenMP_CXX_FOUND)
    target_link_libraries(ml PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(ml PUBLIC -march=native -ffast-math -O3 -fopenmp -lpthread)