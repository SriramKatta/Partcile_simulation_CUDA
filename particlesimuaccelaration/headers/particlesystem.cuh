#pragma once

#include <iostream>
#include <tuple>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

using vec3d = double3;
using hvecofvec = thrust::host_vector<vec3d>;
using hvecofdou = thrust::host_vector<double>;
using dvecofvec = thrust::device_vector<vec3d>;
using dvecofdou = thrust::device_vector<double>;
using hvecofint = thrust::host_vector<int>;
using dvecofint = thrust::device_vector<int>;


struct Particlesystem
{
  Particlesystem() = default;
  void movetoGPU()
  {
    //mdPos.resize(numparticles);
    //mdVel.resize(numparticles);
    //mdAcc.resize(numparticles);
    //mdMass.resize(numparticles);
    //thrust::copy(mhPos.begin(), mhPos.end(), mdPos.begin());
    //thrust::copy(mhVel.begin(), mhVel.end(), mdVel.begin());
    //thrust::copy(mhAcc.begin(), mhAcc.end(), mdAcc.begin());
    //thrust::copy(mhMass.begin(), mhMass.end(), mdMass.begin());

    mdPos = mhPos;
    mdVel = mhVel;
    mdAcc = mhAcc;
    mdMass = mhMass;

    posptr = thrust::raw_pointer_cast(mdPos.data());
    velptr = thrust::raw_pointer_cast(mdVel.data());
    accptr = thrust::raw_pointer_cast(mdAcc.data());
    massptr = thrust::raw_pointer_cast(mdMass.data());
  }

  void initDS(){
    double size = xyzmax - xyzmin;
    cellwidth = std::ceil(cutoffrad);
    nxyzCells = static_cast<int>(size / cellwidth);
    cellcount = nxyzCells * nxyzCells * nxyzCells;
    nxyzCellsStride = {1, nxyzCells, nxyzCells * nxyzCells};
    
    mdCellDS.resize(cellcount);
    mdPartDS.resize(numparticles);

    celldsptr = thrust::raw_pointer_cast(mdCellDS.data());
    partdsptr = thrust::raw_pointer_cast(mdPartDS.data());
  }

  void printDS(){
    hvecofint hveccell = mdCellDS;
    hvecofint hvecpart = mdPartDS;
    for(size_t i=0; i < numparticles; ++i)
      std::cout << hvecpart[i] << " ";
    std::cout << std::endl;
    for(size_t i=0; i < cellcount; ++i)
      std::cout << hveccell[i] << " ";
    std::cout << std::endl;
  }

  std::tuple<vec3d *, vec3d *, vec3d *, double *>
  getDeviceParticlePoints()
  {
    if (posptr == nullptr)
      movetoGPU();
    return {posptr, velptr, accptr, massptr};
  }

  std::tuple<int *, int *>
  getDeviceDSPoints()
  {
    if (celldsptr == nullptr)
      initDS();
    return {celldsptr, partdsptr};
  }
  void synchronize()
  {
    cudaDeviceSynchronize();
    mhPos = mdPos;
    mhVel = mdVel;
    mhAcc = mdAcc;
    //thrust::copy(mdPos.begin(), mdPos.end(), mhPos.begin());
    //thrust::copy(mdVel.begin(), mdVel.end(), mhVel.begin());
    //thrust::copy(mdAcc.begin(), mdAcc.end(), mhAcc.begin());
  }

  void printhstate()
  {
    for (size_t i = 0; i < numparticles; ++i)
    {
      std::cout << " position : " << mhPos[i].x << " " << mhPos[i].y << " "<< mhPos[i].z << " "
                << " velocity : " << mhVel[i].x << " " << mhVel[i].y << " "<< mhVel[i].z << " "
                << " Mass     : " << mhMass[i] << "\n\n";
    }
  }


  void printds()
  {
    thrust::host_vector<int> mhCellDS = mdCellDS;
    thrust::host_vector<int> mhPartDS = mdPartDS;
    std::cout << "cell : ";
    for (auto val : mhCellDS)
    {
      std::cout << val << " ";
    }
    std::cout << "\n";
    std::cout << "part : ";

    for (auto val : mhPartDS)
    {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

  hvecofvec mhPos;
  hvecofvec mhVel;
  hvecofvec mhAcc;
  hvecofdou mhMass;
  size_t ntimesteps;
  double ts;
  double sigma;
  double epsilon;
  double cutoffrad;
  double cellwidth;
  int nxyzCells;
  size_t cellcount;
  int3 nxyzCellsStride;
  double xyzmin;
  double xyzmax;
  size_t numparticles;
  vec3d *posptr = nullptr;
  vec3d *velptr = nullptr;
  vec3d *accptr = nullptr;
  double *massptr = nullptr;
  int* celldsptr = nullptr;
  int* partdsptr = nullptr;

private:
  dvecofvec mdPos;
  dvecofvec mdVel;
  dvecofvec mdAcc;
  dvecofdou mdMass;
  dvecofint mdCellDS;
  dvecofint mdPartDS;
};