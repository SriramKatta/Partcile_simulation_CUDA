#include <chrono>

#include "../util.h"
#include "stream-util.h"

__global__
void stream(size_t nx, const double * src, double * dest) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < nx)
        dest[i] = src[i] + 1;
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);
    //std::cout << "stream parseCLA done" << std::endl;

    double *src,*dest;

    cudaMallocManaged(&src, nx * sizeof(double));
    //std::cout << "stream src alloc done" << std::endl;
    cudaMallocManaged(&dest, nx * sizeof(double));
    //std::cout << "stream dest alloc done" << std::endl;

    // init
    initStream(src, nx);
    //std::cout << "stream init done" << std::endl;

    int threads_per_block = 64;
    int blocks = (threads_per_block + nx -1) / threads_per_block;

    // warm-up
    for (int i = 0; i < nItWarmUp; ++i) {
        stream<<<blocks,threads_per_block>>>(nx, src, dest);
        cudaDeviceSynchronize();
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nIt; ++i) {
        stream<<<blocks,threads_per_block>>>(nx, src, dest);
        cudaDeviceSynchronize();
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);

    // check solution
    checkSolutionStream(src, nx, nIt + nItWarmUp);

    cudaFree(src);
    cudaFree(dest);

    return 0;
}
