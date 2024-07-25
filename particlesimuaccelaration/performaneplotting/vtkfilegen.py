import sys
import numpy as np

def main(numparticles_axis):
  ticks = np.linspace(-numparticles_axis//2,(numparticles_axis + 1)//2, numparticles_axis , dtype=float)
  xx, yy, zz = np.meshgrid(ticks,ticks, ticks)
  x = xx.flatten()
  y = yy.flatten()
  z = zz.flatten()
  print(f"POINTS {len(x)} double")
  for val in zip(x,y,z):
    print(f"{val[0]} {val[1]} {val[2]}")
  a = f"""CELLS 0 0
CELL_TYPES 0
POINT_DATA {len(x)}
SCALARS m double
LOOKUP_TABLE default"""
  print(a)
  for _ in range(len(x)):
    print(1.000)
  
  print("VECTORS v double")
  for _ in range(len(x)):
    print("0.000 0.000 0.000")

if __name__ == "__main__":
  a = """# vtk DataFile Version 4.0 
hesp visualization file 
ASCII 
DATASET UNSTRUCTURED_GRID"""
  print(a)
  main(int(sys.argv[1]))