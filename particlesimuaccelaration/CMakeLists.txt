cmake_minimum_required(VERSION 3.20)

project(MolecularDynamics LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(--expt-relaxed-constexpr -Wall -Wunused) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include(external/thrust.cmake)

option(nbdacc "use nbd acceleration" OFF)
option(cutoff "Use cutoff version " OFF)
option(perfana "Use cutoff version " OFF)

include_directories(${CMAKE_SOURCE_DIR}/headers)
add_subdirectory(${CMAKE_SOURCE_DIR}/implementation)
