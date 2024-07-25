#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <particlesystem.cuh>

void fptrcheck(std::string& fileloc){
    std::ifstream fptr(fileloc);
    if (!fptr.good())
    {
      std::cout << "failure to open the config file" << std::endl;
      fptr.close();
      std::exit(EXIT_FAILURE);
    }
    fptr.close();
}

void parsearguments(int argc, char *argv[],
                    std::string &fileloc,
                    std::string &outfiledir)
{
  if(argc != 3){
    std::cout << "for refrence useage is\n" << argv[0] << " <inputfile> <outputfile> " << std::endl;
    fileloc = "../refrence_input_files/input1";
    outfiledir = "./out/";
    return;
  }
  fileloc = argv[1];
  outfiledir = argv[2];
  fptrcheck(fileloc);
}

