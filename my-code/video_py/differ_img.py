from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(x, y)  # 网格的创建
# Z = (2*X*Y)/(X**2+ Y**2)*(1- (X.any()==0 and Y.any()))+1* (X.any()==0 and Y.any()==0)
Z=(2*X*Y+1)/(X+Y+1)
plt.xlabel('x')
plt.ylabel('y')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()