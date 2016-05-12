import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True, linewidth=200)

###### Iterative closest point (brute-force comparison)

# Given a set of rhos and thetas, find the chessboard lines, and generate missing ones?
rhos = np.array([ 262.        ,   77.81386157,  247.42895939,   88.7388672 ,
              268.        ,  143.70149406,  162.51885098, -158.55033883,
              278.96485925,  197.79203124,  180.65978226,   72.88585375,
              131.75282029,  220.78052799, -156.45596299, -150.82890626,
              100.62710609,  105.80793133, -141.52632746,  195.58986608,
              252.21801208, -127.25451879,  119.63504631,  175.89171612,
              258.        , -118.80304726,  111.91725156])
thetas = np.array([ 1.57079633,  0.2649934 ,  1.5848141 ,  0.18932192,  1.57079633,
        1.60821201,  1.58833839,  2.81240729,  1.55492464,  1.58849359,
        1.58666801,  0.29706421,  1.60430333,  1.58905914,  2.84265183,
        2.91608302,  0.12062367,  1.52007844,  2.98614064,  1.70652062,
        1.5538488 ,  3.01939748,  1.60487404,  1.6026327 ,  1.57079633,
        3.10250642,  0.03844259])


N = rhos.size

# Invert negative rhos
for i in range(N):
  if thetas[i] > .75*np.pi and rhos[i] < 0:
    rhos[i] *= -1
    # thetas[i] = (thetas[i]+np.pi) % np.pi - np.pi
    thetas[i] = (thetas[i]-np.pi)

# Normalize rhos and thetas, based on known sizes
img_maxsize = 320
rhos /= img_maxsize
thetas /= 2*np.pi


print("There are %d given lines" % N)

# theta_mean = thetas.mean()
# rhos_mean = rhos.mean()
# plt.plot([theta_mean, theta_mean], [rhos.min(), rhos.max()],'k--')
# plt.plot(thetas, rhos,'o')
# plt.show()

pos_meas = np.zeros((N,2))
pos_meas[:,0] = thetas
pos_meas[:,1] = rhos

pos_gnd = np.zeros((18,2), dtype=np.float32)
pos_gnd[:9,0] = 0.
pos_gnd[:9,1] = np.arange(9)/9.
pos_gnd[9:,0] = 1.
pos_gnd[9:,1] = np.arange(9)/9.


# Normalize ground readings
pos_gnd = pos_gnd - pos_gnd.mean(0) # center
#pos_gnd = pos_gnd / (pos_gnd.max(0) - pos_gnd.min(0)) # normalize
pos_gnd = pos_gnd * np.sqrt(2)/np.sum(np.sqrt(np.sum(pos_gnd**2,axis=1)))

# Normalize measured readings
pos_meas = pos_meas - pos_meas.mean(0) # center
# pos_meas = pos_meas / (pos_meas.max(0) - pos_meas.min(0)) # normalize
pos_meas = pos_meas * np.sqrt(2)/np.sum(np.sqrt(np.sum(pos_meas**2,axis=1)))

# M_start = np.matrix([[ 0.73126989,-0.25159087,-0.07278123],
#                      [ 0.25159087, 0.73126989,-0.09586837]])

# a = np.ones((18,3))
# a[:,:2] = pos_gnd
#pos_gnd = (M_start * a.T).T.A


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
  print("Iter: %d" % k)
  closest_idxs = getClosestPointsWithRepeats(pos_curr, pos_meas)
  pos_closest = pos_meas[closest_idxs]

  # original points
  #HOMOGRAPHY
  X = np.zeros((2*18,8))
  X[::2,:3] = np.hstack([pos_curr,np.ones([18,1])])
  X[1::2,3:6] = np.hstack([pos_curr,np.ones([18,1])])
  X[::2,6:] = -pos_curr*pos_closest[:,[0,0]]
  X[1::2,6:] = -pos_curr*pos_closest[:,[1,1]]

  #AFFINE
  # X = np.zeros((2*18,6))
  # X[::2,:2] = pos_curr
  # X[::2,2] = 1
  # X[1::2,3:5] = pos_curr
  # X[1::2,5] = 1

  # #ROTATE-TRANSLATE
  # X = np.zeros((2*18,4))
  # X[::2,0:2] = pos_curr
  # X[::2,2] = 1
  # X[1::2,0] = pos_curr[:,1]
  # X[1::2,1] = -pos_curr[:,0]
  # X[1::2,3] = 1

  X = np.mat(X)

  # projected points
  Y = np.ones(2*18)
  Y[::2] = pos_closest[:,0]
  Y[1::2] = pos_closest[:,1]
  Y = Y.reshape(-1,1)
 
  # Least squares estiable coeffs = X \ Y
  coeffs = np.linalg.inv(X.T*X)*X.T*Y

  # Form homography 3x3 matrix
  M =  np.vstack([coeffs,1]).reshape(3,3)

  # new closest pred
  pos_lsq = (M * np.hstack([pos_curr,np.ones((18,1))]).T).T
  pos_lsq /= pos_lsq[:,2]  # normalize by 3rd column
  pos_curr = pos_lsq[:,:2].A 


  plt.cla()
  plt.plot(pos_gnd[:,0],pos_gnd[:,1],'k.',markersize=5,label='gnd')
  plt.plot(pos_lsq[:,0],pos_lsq[:,1],'go',markersize=7,label='lsq')
  plt.plot(pos_meas[:,0],pos_meas[:,1],'rs',markersize=5,label='meas')

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
  plt.pause(0.05)

print(M)
plt.show(block=True)