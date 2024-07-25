#!/bin/bash -l

rm -rf outputvtk/ executable/
if [ $# -ne 0 ]
then
  rm -rf build/
fi