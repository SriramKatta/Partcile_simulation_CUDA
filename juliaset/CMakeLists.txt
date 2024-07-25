cmake_minimum_required(VERSION 3.21)
project(juliaset LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_BUILD_TYPE Release)

add_subdirectory(src)
add_subdirectory(lib)



