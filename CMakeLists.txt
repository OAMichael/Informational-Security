cmake_minimum_required(VERSION 3.13)

project(Informational-Security VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(INCLUDES
    Utils
    3rdparty
)

include_directories(${INCLUDES})

add_subdirectory(BlockCipher)
add_subdirectory(RSA)
