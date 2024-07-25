#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <particlesystem.cuh>

void writeVTKFile(const Particlesystem& system,const  std::string& filename,const std::string& filedir) {
        
        std::filesystem::path vtkDir(filedir);

        if (!std::filesystem::exists(vtkDir)) {
            if (!std::filesystem::create_directories(vtkDir)) {
                std::cerr << "Error: Could not create directory " << vtkDir << std::endl;
                return;
            }
        }
    
        
        std::filesystem::path filePath = vtkDir / filename;
    
        
        std::ofstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filePath << std::endl;
            return;
        }

    size_t N = system.numparticles;

    file << "# vtk DataFile Version 4.0 \n";
    file << "hesp visualization file \n";
    file << "ASCII \n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    file << "POINTS " << N << " double \n";

    for (const auto& pos : system.mhPos) {
        file<< std::fixed << std::setprecision(5) 
            << pos.x << " " 
            << pos.y << " " 
            << pos.z << "\n";
    }

    file << "CELLS 0 0\n";
    file << "CELL_TYPES 0\n";
    file << "POINT_DATA " << N << "\n";
    file << "SCALARS m double\n";
    file << "LOOKUP_TABLE default\n";

    for (const auto& mass : system.mhMass) {
        file<< std::fixed << std::setprecision(5)  << mass << std::endl;
    }

    file << "VECTORS v double" << "\n";

    for (const auto & vel : system.mhVel) {
        file<< std::fixed << std::setprecision(5) 
            << vel.x << " " 
            << vel.y << " " 
            << vel.z << "" << "\n";
    }

    file.close();
}