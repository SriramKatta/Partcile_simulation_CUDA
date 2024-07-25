#pragma once

#include <climits>
#include <cstdlib>
#include <tuple>

using vec3d = double3;
using vec3i = int3;

#define COMPUTESPACE __device__

COMPUTESPACE
vec3d operator+(const vec3d &vec1, const vec3d &vec2)
{
    return {vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z};
}

COMPUTESPACE
vec3d operator+(const vec3d &vec1, const double &val)
{
    return {vec1.x + val, vec1.y + val, vec1.z + val};
}

COMPUTESPACE
vec3d operator-(const vec3d &vec1, const vec3d &vec2)
{
    return {vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z};
}

COMPUTESPACE
vec3d operator-(const vec3d &vec1, const double &val)
{
    return {vec1.x - val, vec1.y - val, vec1.z - val};
}

COMPUTESPACE
vec3d operator*(const vec3d &vec1, const double &val)
{
    return {vec1.x * val, vec1.y * val, vec1.z * val};
}

COMPUTESPACE
vec3d operator/(const vec3d &vec1, const double &val)
{
    return {vec1.x / val, vec1.y / val, vec1.z / val};
}

COMPUTESPACE
vec3i operator%(const vec3i &vec1, const int &val)
{
    return {vec1.x % val, vec1.y % val, vec1.z % val};
}

COMPUTESPACE
vec3i operator*(const vec3i &vec1, const vec3i &vec2)
{
    return {vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z};
}

COMPUTESPACE
double dot(const vec3d &vec1, const vec3d &vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

COMPUTESPACE
vec3i operator+(const vec3i &vec1, const int &val)
{
    return {vec1.x + val, vec1.y + val, vec1.z + val};
}

__device__ double norm(const vec3d &vec)
{
    return norm3d(vec.x, vec.y, vec.z);
}

COMPUTESPACE
void moveintoboundary(vec3d &vec1, const double min, const double max)
{
    auto size = max - min;
    vec1 = vec1 - min;
    vec1.x = fmod(fmod(vec1.x, size) + size, size) + min;
    vec1.y = fmod(fmod(vec1.y, size) + size, size) + min;
    vec1.z = fmod(fmod(vec1.z, size) + size, size) + min;
}

COMPUTESPACE
void reflective_boundary_condition(vec3d &vec1, vec3d &velocity, const double& min, const double& max)
{
    auto reflect = [=](double value, double vel) -> double {
        if (value < min || value >= max) {
            
            return - 1 * vel; 
        }
        return vel; 

    };

    
    velocity.x = reflect(vec1.x, velocity.x);
    velocity.y = reflect(vec1.y, velocity.y);
    velocity.z = reflect(vec1.z, velocity.z);

   
}

COMPUTESPACE
void reflectboundary(vec3d &mpos, vec3d &mvel,const double min, const double max)
{
    auto reflectpos = [=](double pos){
        if(pos > max){
            //vel = - 1 * vel;
            double diff = pos - max;
            pos = pos - (2*diff);
        }
        if(pos < min){
            //vel = - 1 * vel;
            double diff = pos - min;
            pos = pos + (2*diff);
        }
        return pos;
    };
    auto reflectvel = [=](double pos, double vel){
        if(pos > max || pos < min){
            vel = - 0.9 * vel;
        }
        return vel;
    };
    mvel.x = reflectvel(mpos.x, mvel.x);
    mvel.y = reflectvel(mpos.y, mvel.y);
    mvel.z = reflectvel(mpos.z, mvel.z);
    
    mpos.x = reflectpos(mpos.x);
    mpos.y = reflectpos(mpos.y);
    mpos.z = reflectpos(mpos.z);
    
}



__global__ void updatePositions(
    size_t Numpart,
    vec3d *mPosition,
    vec3d *mVelocity,
    vec3d *mAcceleration,
    double xyzmin,
    double xyzmax,
    double dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < Numpart)
    {
        mPosition[i] = mPosition[i] + (mVelocity[i] * dt) + (mAcceleration[i] * (0.5 * dt * dt));
        reflective_boundary_condition(mPosition[i],mVelocity[i], xyzmin, xyzmax);
    }
}

__global__ void updateVelocitiesHalfStep(
    size_t Numpart,
    vec3d *mVelocity,
    vec3d *mAcceleration,
    double dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < Numpart)
    {
        mVelocity[i] = mVelocity[i] + (mAcceleration[i] * (0.5 * dt));
    }
}

__global__ void resetDS(
    size_t Numpart,
    size_t Ncells,
    int *cellDSPtr,
    int *partDSPtr)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < Ncells)
    {
        cellDSPtr[index] = -1;
    }
    if (index < Numpart)
    {
        partDSPtr[index] = index;
    }
}

__device__
    vec3i
    cellindexcalc(const vec3d &pos, const int &ncells, const double &min, const double &max)
{
    const auto size = max - min;
    const auto ticks = size / static_cast<double>(ncells);
    auto normpos = pos - min;
    vec3i cellind;
    cellind.x = floor(normpos.x / ticks);
    cellind.y = floor(normpos.y / ticks);
    cellind.z = floor(normpos.z / ticks);

    return cellind;
    // auto size = xyzmax - xyzmin; //20
    // const vec3d ticks = size.cwiseQuotient(nxyzCells.cast<double>()); //
    // const vec3d normparticlepos = mPosition[index] - xyzmin;
    // normparticlepos.cwiseQuotient(ticks).array().cast<int>();
}

COMPUTESPACE
int lincellindex(const vec3i &cellindex, const int &ncells, const vec3i &stride)
{
    auto temp = ((cellindex + ncells) % ncells) * stride; // normalization also included
    return temp.x + temp.y + temp.z;
    // for (int l = 0; l < 3; ++l)
    //     cell += (cellindex[l] % nxyzCells[l]) * nxyzCellsStride[l];
}

__global__ void updateDS(
    size_t Numpart,
    size_t Ncells,
    vec3d *mPosition,
    double xyzmin,
    double xyzmax,
    int nxyzCells,
    vec3i nxyzCellsStride,
    int *cellDSPtr,
    int *partDSPtr)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < Numpart)
    {
        auto cellindex = cellindexcalc(mPosition[index], nxyzCells, xyzmin, xyzmax);
        int cell = lincellindex(cellindex, nxyzCells, nxyzCellsStride);
        partDSPtr[index] = atomicExch(&cellDSPtr[cell], index);
    }
}

__device__
    vec3d
    pbcdistvectcalc(const vec3d xij, const double &size)
{
    // xij - size.cwiseProduct(xij.cwiseQuotient(size).array().round().matrix());
    vec3d subterm;
    subterm.x = round(__ddiv_rd(xij.x, size));
    subterm.y = round(__ddiv_rd(xij.y, size));
    subterm.z = round(__ddiv_rd(xij.z, size));
    return xij - (subterm * size);
}

COMPUTESPACE inline
double HSstep(const double& x ){
    return (x>=0) ? 1.0 : 0.0;
}


COMPUTESPACE inline
vec3d XI(const double& sigma, const vec3d& xij){
    const auto xijnorm = norm(xij);
    const auto intterm = sigma - xijnorm;
    const auto scaleterm = intterm * HSstep(intterm) / xijnorm;
    return xij * scaleterm;
}


COMPUTESPACE inline
vec3d XIdot(const double& sigma, const vec3d& xij, const vec3d& vij){
    const auto xijnorm = norm(xij);
    const auto xijcap = xij / xijnorm;
    const auto intterm = sigma - xijnorm;
    const auto scaleterm = -1 * dot(xijcap, vij) * HSstep(intterm);
    return xijcap * scaleterm;
}


COMPUTESPACE
vec3d forcecalc(const double &springK, const double &springC, const double& sigma, const vec3d &xij, const vec3d& vij)
{
    return XI(sigma, xij) * springK + XIdot(sigma, xij, vij) * springC;
}

__global__ void updateAcceleration(
    size_t N,
    vec3d *mPosition,
    vec3d *mVelocity,
    vec3d *mAcceleration,
    double *mMass,
    double *mRadius,
    double springK,
    double springC,
    double gravity,
    double xyzmin,
    double xyzmax,
    int nxyzCells,
    vec3i nxyzCellsStride,
    int *cellDSPtr,
    int *partDSPtr,
    double dt)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
    {
        const auto positioni = mPosition[index];
        const auto velocityi = mVelocity[index];
        const auto radiusi = mRadius[index];
        vec3i cellloc = cellindexcalc(positioni, nxyzCells, xyzmin, xyzmax);
        vec3d f = {0.0, 0.0, 0.0};
        for (int i = cellloc.x - 1; i <= cellloc.x + 1; ++i)
        {
            for (int j = cellloc.y - 1; j <= cellloc.y + 1; ++j)
            {
                for (int k = cellloc.z - 1; k <= cellloc.z + 1; ++k)
                {
                    vec3i cellnum = {i, j, k};
                    int cellind = lincellindex(cellnum, nxyzCells, nxyzCellsStride);
                    int partindex = cellDSPtr[cellind];
                    while (partindex != -1)
                    {
                        if (partindex != index)
                        {
                            vec3d xij = positioni - mPosition[partindex];
                            double sigma = radiusi + mRadius[partindex];
                            auto dist = norm(xij);
                            if (HSstep(sigma - dist) == 1)
                            {
                                vec3d vij = velocityi - mVelocity[partindex];
                                const auto fsd = forcecalc(springK, springC, sigma, xij, vij);
                                f = f + fsd;
                            }
                        }
                        partindex = partDSPtr[partindex];
                    }
                }
            }
        }
        mAcceleration[index] = f / mMass[index];
        mAcceleration[index].z += gravity;
    }
}

__global__ void updateVelocitiesFullStep(
    size_t N,
    vec3d *mVelocity,
    vec3d *mAcceleration,
    double dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        mVelocity[i] = mVelocity[i] + (mAcceleration[i] * (0.5 * dt));
    }
}

double normsquaredhost(const vec3d &vec)
{
    return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

double normhost(const vec3d &vec)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

double calculateTotalEnergy(
    const thrust::host_vector<vec3d>& mPosition,
    const thrust::host_vector<vec3d>& mVelocity,
    const thrust::host_vector<double>& mMass,
    const double &gravity,
    const double &xyzmin)
{
    double totalEnergy = 0.0;
    size_t N = mPosition.size();
    for (size_t i = 0; i < N; ++i)
    {
        totalEnergy += 0.5 * mMass[i] * normsquaredhost(mVelocity[i]);
        totalEnergy += mMass[i] * gravity * (mPosition[i].z - xyzmin);
    }
    return totalEnergy;
}

/*
__global__ void updateAccelerationsimple(
    size_t N, vec3d *mPosition,
    vec3d *mAcceleration,
    double *mMass,
    double sigma,
    double epsilon,
    double cutoffrad,
    double xyzmin,
    double xyzmax,
    double dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        vec3d f = {0.0, 0.0, 0.0};
        auto particlei = mPosition[i];
        auto size = xyzmax - xyzmin;
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                auto xij = particlei - mPosition[j];
                auto xijpbc = pbcdistvectcalc(xij, size);
                auto dist = norm(xijpbc);
                if (dist < cutoffrad)
                {
                    const auto force = forcecalc(sigma, epsilon, dist);
                    f = f + (xijpbc * force);
                }
            }
        }
        mAcceleration[i] = f / mMass[i];
    }
}

__global__ void updateAccelerationbase(
    size_t N, vec3d *mPosition,
    vec3d *mAcceleration,
    double *mMass,
    double sigma,
    double epsilon,
    double dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        vec3d f = {0.0, 0.0, 0.0};
        auto particlei = mPosition[i];
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                auto xij = particlei - mPosition[j];
                auto dist = norm(xij);
                const auto force = forcecalc(sigma, epsilon, dist);
                f = f + (xij * force);
            }
        }
        mAcceleration[i] = f / mMass[i];
    }
}



if (index < N)
    {
        vec3i cellindex = cellindexcalc(mPosition[index], nxyzCells, xyzmin, xyzmax);
        for (int i = cellindex.x - 1; i <= cellindex.x + 1; ++i)
            for (int j = cellindex.y - 1; j <= cellindex.y + 1; ++j)
                for (int k = cellindex.z - 1; k <= cellindex.z + 1; ++k)
                {
                    int cell = lincellindex({i, j, k}, nxyzCellsStride);
                    vec3d f = {0.0, 0.0, 0.0};
                    int particleindex = cellDSPtr[cell];
                    while (particleindex != -1)
                    {
                        if (particleindex != index)
                        {
                            auto xij = mPosition[index] - mPosition[particleindex];
                            auto xijpbc = pbcdistvectcalc(xij, xyzmax - xyzmin);
                            double dist = norm(xijpbc);
                            if (dist < cutoffrad)
                            {
                                const double sbdist = sigma / dist;
                                const double force = 24 * epsilon * pow(sbdist, 6) * ((2 * pow(sbdist, 6)) - 1) / (dist * dist);
                                f = f + xijpbc * force;
                            }
                        }
                        particleindex = partDSPtr[particleindex];
                    }
                    mAcceleration[index] = f / mMass[index];
                }
    }

*/