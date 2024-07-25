#!/bin/bash -l

cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release
cmake --build build/ -j

./executable/MDsimu 25000 0.01 2 1 refrence_input_files/inputref1 vtk/1
./executable/MDsimu 25000 0.01 2 1 refrence_input_files/inputref2 vtk/2
./executable/MDsimu 25000 0.01 .91 -1 ./refrence_input_files/inputref3 vtk/3