cmake_minimum_required(VERSION 3.13)

project(Informational-Security VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Benchmark build
add_executable(Benchmark Benchmark.cpp) 

include_directories(Testing)

# Directories with siphers
add_subdirectory(BlockCipher)
add_subdirectory(Mock)