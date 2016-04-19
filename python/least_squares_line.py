import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

####### Line fit
N=50
x = np.linspace(-13,23,N)
y_ground = 0.23*x - 12
y = y_ground + np.random.randn(N)*2


X = np.ones((N,2))
X[:,0] = x

X = np.mat(X)


Y = y.reshape(-1,1)

coeffs = np.linalg.inv(X.T*X)*X.T*Y

print(coeffs)

y_lsq = X*coeffs


plt.plot(x,y_ground,'k--',label='ground')
plt.plot(x,y,'r.',label='data')
plt.plot(x,y_lsq,'b',label='least squares')
plt.legend(loc='best')
plt.show()