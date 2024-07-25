import sys
import numpy as np

def main(xyzlimitsabs):
  ticks = np.arange(-xyzlimitsabs+0.5, xyzlimitsabs, 2,dtype=float)
  xx, yy,zz = np.meshgrid(ticks,ticks,ticks)
  x = xx.flatten()
  y = yy.flatten()
  z = zz.flatten()
  print(len(x))
  print("POINTS")
  for val in zip(x,y,z):
    print(val[0],  val[1], val[2])
  print("MASS")
  
  for _ in range(len(x)):
    print(1.000)
  
  print("VELOCITY")
  for val in zip(x,y,z):
    print("0.0 0.0 0.0")
  
  print("RADIUS")
  for _ in range(len(x)):
    print("1.0")

if __name__ == "__main__":
  xyzlimitsabs = int(sys.argv[3])
  springk = 50#0.01 + (1/(2**(1/6)))
  springc = 25
  gravity = -9.8
  print(sys.argv[1])
  print(sys.argv[2])
  print(gravity)
  print(springk)
  print(springc)
  print(-xyzlimitsabs-2)
  print(xyzlimitsabs+2)
  main(xyzlimitsabs)