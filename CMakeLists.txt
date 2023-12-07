cmake_minimum_required(VERSION 3.13)

project(Informational-Security VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(COMPILE_OPT -O3)
add_compile_options(${COMPILE_OPT})


set(INCLUDES
    Utils
    3rdparty
    Ciphers
)

include_directories(${INCLUDES})

# Benchmark build
add_executable(Benchmark Benchmark.cpp) 
set_target_properties(Benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include_directories(Testing)

add_subdirectory(Ciphers)
