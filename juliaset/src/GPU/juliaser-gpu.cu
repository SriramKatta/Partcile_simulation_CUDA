#include <iostream>
#include <complex>
#include <vector>
#include <lodepng.h>
#include <chrono>

using Time = std::chrono::high_resolution_clock;

__global__
void imagesynth(unsigned char *resultimage, int resx, int resy)
{
  int max_iterations = 255;
  double escape_radius = 2.0;
  double c_real = -0.2;
  double cimg = 0.81;
  int iteration = 0;
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(y < resx && x < resy)
  {
    
    size_t index = 4 * (resx * y +  x);
    double zreal = static_cast<double>(x - resx / 2) / resx;
    double zimg = static_cast<double>(y - resy / 2) / resy;
    resultimage[index + 1] = zreal;
    resultimage[index + 2] = zimg;
    resultimage[index + 3] = 255;
    while ((zreal * zreal + zimg * zimg) <= escape_radius && iteration < max_iterations)
    {
      double zrealtemp = zreal;
      zreal = zreal * zreal - zimg * zimg + c_real;
      zimg = zimg * zrealtemp * 2.0 + cimg;
      ++iteration;
    }
    resultimage[index] = iteration;
  }
}

void encodepng(const char *filename, unsigned char *image, unsigned width, unsigned height)
{
  // Encode the image
  unsigned error = lodepng::encode(filename, image, width, height);

  // if there's an error, display it
  if (error)
    std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

int main(int argc, char const *argv[])
{
  int resx = 4 * 1024;
  int resy = resx;
  size_t datasize = resx * resy * 4 * sizeof(unsigned char);
  unsigned char *resimage;
  cudaMallocManaged(&resimage , datasize);
  
  cudaError_t err = cudaGetLastError();
   if(err != cudaSuccess)
    printf("1 Error is %s\n", cudaGetErrorString(err));

  dim3 threads_per_block(32, 32, 1);
  dim3 number_of_blocks((resx/threads_per_block.x) + 1, (resx / threads_per_block.y) + 1, 1);

  cudaDeviceSynchronize();  
auto start = Time::now();
  imagesynth <<< number_of_blocks, threads_per_block >>> (resimage, resx, resy);
  
  err = cudaGetLastError();
  if(err != cudaSuccess)
    printf("2 Error is : %s\n", cudaGetErrorString(err));
  
  err = cudaDeviceSynchronize();
auto end = Time::now();
  if(err != cudaSuccess)
    printf("3 Error is %s\n", cudaGetErrorString(err));

  encodepng("image_gpu.png", resimage, resx, resy);

  std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  cudaFree(resimage);

  return 0;
}
