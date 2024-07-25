import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate the data
ticks = np.arange(-2, 3)
xx, yy, zz = np.meshgrid(ticks, ticks, ticks)

# Flatten the arrays
x = xx.flatten()
y = yy.flatten()
z = zz.flatten()

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# Labeling the axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Set the title
ax.set_title('3D Scatter Plot')

plt.show()
