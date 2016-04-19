import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

####### Point fit Affine
N=20

# Original
x0 = np.random.random(N)*20
y0 = np.random.random(N)*20

x_ground = 3*x0 + 20
y_ground = 10*y0 + 34

# Projected
x = x_ground + np.random.randn(N)
y = y_ground + np.random.randn(N)


# original points
X = np.zeros((2*N,6))
X[::2,0] = x0
X[::2,1] = y0
X[::2,2] = 1
X[1::2,3] = x0
X[1::2,4] = y0
X[1::2,5] = 1
X = np.mat(X)

# proj = M * trans
# trans = inv(M) * proj

# projected points
Y = np.ones(2*N)
Y[::2] = x
Y[1::2] = y
Y = Y.reshape(-1,1)


coeffs = np.linalg.inv(X.T*X)*X.T*Y
M = coeffs.reshape(2,3)
print(M)

# warped points
pts = np.vstack([x0, y0, np.ones(N)])
pts_lsq = M * pts
x_lsq = pts_lsq[0,:].A1
y_lsq = pts_lsq[1,:].A1


plt.plot(x0,y0,'k.',label='original')
# plt.plot(x_ground,y_ground,'k.',label='ground')
plt.plot(x,y,'rh',markersize=10,label='data')
plt.plot(x_lsq, y_lsq,'b.',label='least squares')
plt.axis('equal')
plt.legend(loc='best')
plt.show()