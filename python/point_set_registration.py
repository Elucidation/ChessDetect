import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True, linewidth=200)

###### Iterative closest point (brute-force comparison)
N=7

k = np.tile(np.arange(N, dtype=np.float32),(N,1))
pos_gnd = np.vstack([k.flatten(), k.flatten('F')]).T

# normalize pos_gnd
pos_gnd = pos_gnd - pos_gnd.mean(0) # center
pos_gnd = pos_gnd / (pos_gnd.max(0) - pos_gnd.min(0)) # normalize


# Actual affine transformation matrix
rot = 13*np.pi/180
M_real = np.mat([
  [np.cos(rot)*0.8, -np.sin(rot), 1],
  [np.sin(rot), np.cos(rot)*0.9, 32],
  [0,0,1]
  ])

print(M_real)

# Ground as a Nx3 vector (added 1's column for affine transformation multiply)
gnd_vec = np.hstack([pos_gnd,np.ones((N*N,1))])

pos_meas = (M_real * gnd_vec.T).T
pos_meas = pos_meas[:,:2].A # remove 1's column and back to array

# shuffle all measured readings since we don't know what order it'd be in
np.random.shuffle(pos_meas)

# Normalize measured readings
pos_meas = pos_meas - pos_meas.mean(0) # center
pos_meas = pos_meas / (pos_meas.max(0) - pos_meas.min(0)) # normalize


def findClosestPoint(pos, arr):
  """Returns the index in pos arr (shape Nx2) with point closest to pos vector"""
  best_dist = None
  best_i = 0
  # Return index of smallest distance (brute force comparison)
  return np.nanargmin(np.sum((arr - pos)**2, 1))

def getClosestPointsWithRepeats(arr1, arr2):
  """Return indices of closest points in arr2 to points in arr1,
  number of indices = len(arr1), allows repeats"""

  return np.array([findClosestPoint(pos, arr2) for pos in arr1])

def getClosestPoints(arr1, arr2):
  """Return indices of closest points in arr2 to points in arr1,
  number of indices = len(arr1), allows repeats, no repeated indices"""

  # Naively choose closest point, replace that pos with a NaN value so no repeats
  temp_arr = arr2.copy()

  best_indices = np.zeros(arr1.shape[0], dtype=np.int)
  for i in range(arr1.shape[0]):
    best_indices[i] = findClosestPoint(arr1[i], temp_arr)
    temp_arr[best_indices[i],:] = None
  return best_indices


plt.ion()
plt.show()

pos_curr = pos_gnd.copy()
for k in range(10):


  closest_idxs = getClosestPointsWithRepeats(pos_curr, pos_meas)
  pos_closest = pos_meas[closest_idxs]
  print(closest_idxs)

  # original points
  X = np.zeros((2*N*N,6))
  X[::2,:2] = pos_curr
  X[::2,2] = 1
  X[1::2,3:5] = pos_curr
  X[1::2,5] = 1
  X = np.mat(X)

  # projected points
  Y = np.ones(2*N*N)
  Y[::2] = pos_closest[:,0]
  Y[1::2] = pos_closest[:,1]
  Y = Y.reshape(-1,1)


  coeffs = np.linalg.inv(X.T*X)*X.T*Y
  M = coeffs.reshape(2,3)
  # print(M)

  if (M[0,0] + M[1,1] == 0):
    print("Singular")

  # new closest pred
  pos_lsq = (M * np.hstack([pos_curr,np.ones((N*N,1))]).T).T
  pos_lsq = pos_lsq[:,:2].A
  pos_curr = pos_lsq


  plt.cla()
  # plt.plot(pos_gnd[:,0],pos_gnd[:,1],'kh',markersize=1,label='gnd')
  plt.plot(pos_meas[:,0],pos_meas[:,1],'r.',markersize=10,label='gnd')
  plt.plot(pos_lsq[:,0],pos_lsq[:,1],'gh',markersize=10,label='lsq')

  # Plot lines to closest matches
  for i in range(pos_gnd.shape[0]):
    plt.plot(
      [pos_curr[i,0], pos_meas[closest_idxs[i],0]],
      [pos_curr[i,1], pos_meas[closest_idxs[i],1]],
      'b:')


  plt.axis('equal')
  plt.legend(loc='best')
  plt.title('Iter: %d' % k)
  plt.draw()
  plt.pause(0.1)

plt.show(block=True)
