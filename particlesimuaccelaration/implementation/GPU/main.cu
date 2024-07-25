#include <particlesystem.cuh>
#include <configreader.cuh>
#include <parseCLAarguments.cuh>
#include <computehelper.cuh>
#include <vtkwriter.cuh>
#include <chrono>

using namespace std::chrono;

void setgpuconfig(int numelements, int& threadspblk, int& blocks, int numberOfSMs){
    threadspblk = 32 * 8; //256
    blocks = numberOfSMs * ((numelements / (threadspblk * numberOfSMs ))+1) ;
}

int main(int argc, char *argv[])
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  Particlesystem system;
  std::string fileloc = "../refrence_input_files/input1";
  std::string outfiledir = "./outdir/";

  parsearguments(argc, argv, fileloc, outfiledir);

  std::cout << fileloc << "\n" << outfiledir << "\n";

  creader(system, fileloc);

  print_values(system, fileloc, outfiledir);

  auto [posptr, velptr, accptr, massptr] = system.getDeviceParticlePoints();
#if nbdacc
  auto [cellptr, partptr] = system.getDeviceDSPoints();
#endif
  int thpart, blks;
  int maxval = std::max(system.numparticles, system.cellcount);
  setgpuconfig(maxval, thpart, blks, numberOfSMs); 

#if perfana
  std::cout << "perfana mode" << std::endl;
  auto start = high_resolution_clock::now();
#endif
  for (size_t step = 0; step <= system.ntimesteps; ++step)
  {
    updatePositions<<<blks, thpart>>>(system.numparticles, posptr, velptr, accptr, system.xyzmin, system.xyzmax,system.ts);
    updateVelocitiesHalfStep<<<blks, thpart>>>(system.numparticles, velptr, accptr, system.ts);
    #if nbdacc
    resetDS<<<blks, thpart>>>(system.numparticles, system.cellcount, cellptr, partptr);
    updateDS<<<blks, thpart>>>(system.numparticles, system.cellcount, posptr, system.xyzmin, system.xyzmax, system.nxyzCells, system.nxyzCellsStride, cellptr, partptr);
    updateAcceleration<<<blks, thpart>>>(system.numparticles, posptr, accptr, massptr, system.sigma, system.epsilon, system.cutoffrad, system.xyzmin, system.xyzmax, system.nxyzCells, system.nxyzCellsStride, cellptr, partptr, system.ts);
    #elif cutoff
    updateAccelerationsimple<<<blks, thpart>>>(system.numparticles, posptr, accptr, massptr, system.sigma, system.epsilon, system.cutoffrad, system.xyzmin, system.xyzmax, system.ts);
    #else
    updateAccelerationbase<<<blks, thpart>>>(system.numparticles, posptr, accptr, massptr, system.sigma, system.epsilon, system.ts);
    #endif
    updateVelocitiesFullStep<<<blks, thpart>>>(system.numparticles, velptr, accptr, system.ts);
#if perfana
  }
  system.synchronize();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end - start);
  std::cout << system.numparticles << " " << static_cast<double>(duration.count())/system.ntimesteps << std::endl;
#else
    if(step % 100 == 0 || step == system.ntimesteps){
      system.synchronize();
      double totalEnergy = calculateTotalEnergy(system.mhPos, system.mhVel, system.mhMass, system.sigma, system.epsilon, system.cutoffrad);
      std::cout << "Step " << step << ", Total Energy: " << totalEnergy << "\n";
      std::string fname("output");
      fname += std::to_string(step);
      fname += ".vtk";
      writeVTKFile(system, fname, outfiledir);
    }
  }
#endif
  return 0;
}


/*
  #if performanceplot
  }
  system.synchronize();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end - start);
  std::cout << system.numparticles << " " << static_cast<double>(duration.count())/ntimesteps << std::endl;
  #else

  #endif

*/