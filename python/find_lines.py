import numpy as np
from matplotlib import pyplot as plt
# from sklearn import linear_model, datasets
np.set_printoptions(suppress=True)

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

n = rhos.size
print("There are %d given lines" % n)
print("rhos\n", rhos)
print("thetas\n", np.floor(thetas*180/np.pi))

# Invert negative rhos
for i in range(n):
  if thetas[i] > .75*np.pi and rhos[i] < 0:
    rhos[i] *= -1
    # thetas[i] = (thetas[i]+np.pi) % np.pi - np.pi
    thetas[i] = (thetas[i]-np.pi)


theta_mean = thetas.mean()
# plt.plot(thetas*180/np.pi, rhos,'rs')
plt.plot([theta_mean*180/np.pi, theta_mean*180/np.pi], [rhos.min(), rhos.max()],'k--')

setsplit = thetas < theta_mean

thetas1 = thetas[setsplit]
thetas2 = thetas[~setsplit]

rhos1 = rhos[setsplit]
rhos2 = rhos[~setsplit]

print("RANSAC start")
# Robustly fit linear model with RANSAC algorithm
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), min_samples=0.3)
model_ransac2 = linear_model.RANSACRegressor(linear_model.LinearRegression(), min_samples=0.3)
X1 = thetas1.reshape(-1,1)
y1 = rhos1.reshape(-1,1)
X2 = thetas2.reshape(-1,1)
y2 = rhos2.reshape(-1,1)



model_ransac.fit(X1, y1)
model_ransac2.fit(X2, y2)
print("RANSAC end")

inlier_mask1 = model_ransac.inlier_mask_
outlier_mask1 = np.logical_not(inlier_mask1)

inlier_mask2 = model_ransac2.inlier_mask_
outlier_mask2 = np.logical_not(inlier_mask2)


# plt.plot(thetas1*180/np.pi, rhos1,'rs')
# plt.plot(thetas2*180/np.pi, rhos2,'bs')
plt.plot(X1[inlier_mask1]*180/np.pi, y1[inlier_mask1], 'gs', label='Inliers1')
plt.plot(X1[outlier_mask1]*180/np.pi, y1[outlier_mask1], 'ys', label='Outliers2')
plt.plot(X2[inlier_mask2]*180/np.pi, y2[inlier_mask2], 'go', label='Inliers2')
plt.plot(X2[outlier_mask2]*180/np.pi, y2[outlier_mask2], 'yo', label='Outliers2')

plt.show()

