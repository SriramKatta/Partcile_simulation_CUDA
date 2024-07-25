#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <tuple>

#include <particlesystem.cuh>
#include <thrust/host_vector.h>

using ptype = std::ifstream::pos_type;
using vec3d = double3;

void print_values(Particlesystem &system, std::string fileloc, std::string outfiledir)
{
  std::cout << "time step : " << system.ts << " "
            << "\nnumber of time steps : " << system.ntimesteps << " "
            << "\nspring K value : " << system.springK << " "
            << "\ndamper C value : " << system.springC << " "
            << "\ninput file loc : " << fileloc << " "
            << "\noutputfolder loc : " << outfiledir << " "
            //<< "\ncutoff radius value : " << system.cutoffrad << " "
            << "\nxyzmin value : " << system.xyzmin << " "
            << "\nxyzmin value : " << system.xyzmax << " "
            << "\nnumparticles : " << system.numparticles << std::endl;
}

void getdatapos(std::ifstream &fptr, ptype &points_pos, ptype &mass_pos, ptype &vel_pos, ptype &rad_pos)
{
  std::string line;
  while (std::getline(fptr, line))
  {
    if (line == "POINTS")
      points_pos = fptr.tellg();
    else if (line == "MASS")
      mass_pos = fptr.tellg();
    else if (line == "VELOCITY")
      vel_pos = fptr.tellg();
    else if (line == "RADIUS")
      rad_pos = fptr.tellg();
  }
}

vec3d readvecfromfptr(std::ifstream &fptr)
{
  double val1, val2, val3;
  fptr >> val1 >> val2 >> val3;
  return {val1, val2, val3};
}

double readscalarfromfptr(std::ifstream &fptr)
{
  double val;
  fptr >> val;
  return val;
}

void creader(Particlesystem& system , std::string fileloc)
{
  std::ifstream fptr(fileloc);
  ptype points_pos, mass_pos, vel_pos, rad_pos;
  fptr >> system.ntimesteps;
  fptr >> system.ts;
  fptr >> system.gravity;
  fptr >> system.springK;
  fptr >> system.springC;
  fptr >> system.xyzmin;
  fptr >> system.xyzmax;
  fptr >> system.numparticles;
  system.mhPos.resize(system.numparticles);
  system.mhVel.resize(system.numparticles);
  system.mhAcc.resize(system.numparticles);
  system.mhMass.resize(system.numparticles);
  system.mhRadius.resize(system.numparticles);
  getdatapos(fptr, points_pos, mass_pos, vel_pos, rad_pos);
  std::ifstream fptrpoints(fileloc), fptrmass(fileloc), fptrvel(fileloc), fptrradius(fileloc);
  fptrpoints.seekg(points_pos);
  fptrmass.seekg(mass_pos);
  fptrvel.seekg(vel_pos);
  fptrradius.seekg(rad_pos);

  for (size_t i = 0; i < system.numparticles; ++i){
    system.mhPos[i] = readvecfromfptr(fptrpoints);
    system.mhVel[i] = readvecfromfptr(fptrvel);
    system.mhAcc[i] = {0.0, 0.0, 0.0};
    system.mhMass[i] = readscalarfromfptr(fptrmass);
    system.mhRadius[i] = readscalarfromfptr(fptrradius);
  }
}