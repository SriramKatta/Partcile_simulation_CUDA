import sys
import numpy as np

def main(xyzlimitsabs):
  ticks = np.arange(-xyzlimitsabs+0.5, xyzlimitsabs, dtype=float)
  xx, yy,zz = np.meshgrid(ticks,ticks,ticks)
  x = xx.flatten()
  y = yy.flatten()
  z = zz.flatten()
  print(len(x))
  print("POINTS")
  for val in zip(x,y,z):
    print(f"{val[0]} {val[1]} {val[2]}")
  print("MASS")
  
  for i in range(len(x)):
    print(1.000)
  
  print("VELOCITY")
  for val in zip(x,y,z):
    print(f"{-val[0]/xyzlimitsabs**2} {-val[1]/xyzlimitsabs**2} {-val[2]/xyzlimitsabs**2}")

if __name__ == "__main__":
  xyzlimitsabs = int(sys.argv[3])
  sigma = 0.35#0.01 + (1/(2**(1/6)))
  epsilon = -10
  cutfoffrad = 2.52
  print(sys.argv[1])
  print(sys.argv[2])
  print(sigma)
  print(epsilon)
  print(cutfoffrad)
  print(-xyzlimitsabs-1)
  print(xyzlimitsabs+1)
  main(xyzlimitsabs)