#!/bin/bash
dldir="build"

[ ! -d "$dldir" ] && mkdir -p "$dldir"

cd "$dldir"
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ..
./executable/juliasetcpu
./executable/juliasetgpu

