#include <iostream>
#include <complex>
#include <vector>
#include <lodepng.h>
#include <chrono>

using Time = std::chrono::high_resolution_clock;

void imagesynth(unsigned char *resultimage, int resx, int resy)
{
  int max_iterations = 255;
  double escape_radius = 2.0;
  double c_real = -0.2;
  double cimg = 0.81;
  for (int y = 0; y < resx; ++y)
  for (int x = 0; x < resy; ++x)
  {
    int iteration = 0;
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
    resultimage[index + 0] = iteration;
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
  size_t datasize = resx * resy * 4;
  unsigned char *resimage = new unsigned char[datasize];
  auto start = Time::now();
  imagesynth(resimage, resx, resy);
  auto end = Time::now();
  encodepng("image_cpu.png", resimage, resx, resy);
  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
  delete[] resimage;
  return 0;
}
