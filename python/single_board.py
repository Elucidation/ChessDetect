import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import find_regularly_spaced
np.set_printoptions(suppress=True)


chessboard_idx = 4

# Load image
filename = 'chessboard%d.jpg' % chessboard_idx
print(filename)

original_img = cv2.imread(filename)
print("Original Shape [y, x, channels]: ",original_img.shape)

# Get scaled image
img_max_width = 512
img_scale_ratio = img_max_width / original_img.shape[1]

img = cv2.resize(original_img, (0,0), fx=img_scale_ratio, fy=img_scale_ratio)

# Get Grayscale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Processed Shape [y, x]: ",gray_img.shape)


# Get Canny Edges
# edges = cv2.Canny(cv2.blur(gray_img, (3,3)), 250, 256, apertureSize=3)
edges = cv2.Canny(gray_img, 250, 256, apertureSize=3)


# Get hough lines
# lines = cv2.HoughLines(edges,1,np.pi/180,50)
lines = cv2.HoughLinesP(edges,1,np.pi/180, 50, minLineLength=30, maxLineGap=200)

if lines is None:
  raise("No Lines found!")
else:
  print("Number of lines found:",len(lines))


thetas = []
rhos = []

# Collect thetas
line_subset = np.ones(lines.shape[0], dtype=bool)
i=0
for x1,y1,x2,y2 in lines[:,0,:]:
  # x1*cos(theta) + y1*sin(theta) = rho = x2*cos(theta) + y2*sin(theta)
  # x1 + y1*sin(theta)/cos(theta) = x2 + y2*sin(theta)/cos(theta)
  # x1 + y1*tan(theta) = x2 + y2*tan(theta)
  # (y1 - y2)*tan(theta) = x2 - x1
  # tan(theta) = (x2 - x1) / (y1 - y2)
  # theta = atan2((x2 - x1), (y1 - y2))
  theta = np.math.atan2((x2 - x1), (y1 - y2))
  rho = x1*np.cos(theta) + y1*np.sin(theta)

  # If near one of the other values, ignore this one
  is_duplicate = False
  for other_theta in thetas:
    # if within 1.2 degrees
    if np.abs(theta - other_theta) < 1.2*np.pi/180.0:
      for other_rho in rhos:
        # if within 6 pixels line normal displacement
        if np.abs(rho - other_rho) < 6:
          is_duplicate = True
          break
    if is_duplicate:
      break

  if is_duplicate:
    line_subset[i] = False
  else:
    line_subset[i] = True

  thetas.append(theta)
  rhos.append(rho)
  i += 1

# Set of all lines, thetas and rhos
all_lines = lines.copy()
all_thetas = np.array(thetas)
all_rhos = np.array(rhos)

# Only keep those lines that weren't too close to others
lines = all_lines[line_subset,:,:]
thetas = all_thetas[line_subset]
rhos = all_rhos[line_subset]

# Invert negative rhos where theta > 135 deg
for i in range(rhos.size):
  if thetas[i] > .75*np.pi and rhos[i] < 0:
    rhos[i] *= -1
    # thetas[i] = (thetas[i]+np.pi) % np.pi - np.pi
    thetas[i] = (thetas[i]-np.pi)

print("Number of pruned lines found:",len(rhos))
# plt.plot(thetas*180/np.pi,rhos,'rs')

# Keep pruned set
pruned_lines = lines.copy()
pruned_thetas = thetas.copy()
pruned_rhos = rhos.copy()

# RANSAC find valid lines for chessboard only
theta_mean = thetas.mean()

setsplit = thetas < theta_mean

thetas1 = thetas[setsplit]
thetas2 = thetas[~setsplit]

rhos1 = rhos[setsplit]
rhos2 = rhos[~setsplit]

print("RANSAC start")
# Robustly fit linear model with RANSAC algorithm
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(normalize=True))
model_ransac2 = linear_model.RANSACRegressor(linear_model.LinearRegression(normalize=True))

# Since we can have vertical lines with (x,y) = (theta,rho), but never horizontal, and
# linear regression has difficulty with vertical lines, we can flip the two, essentially
# solving for x = ny + b instead of y = mx+b
X1 = thetas1.reshape(-1,1)
y1 = rhos1.reshape(-1,1)
X2 = thetas2.reshape(-1,1)
y2 = rhos2.reshape(-1,1)

model_ransac.fit(X1, y1)
model_ransac2.fit(X2, y2)
print("RANSAC end")

# Get valid masks for each set
inlier_mask1 = model_ransac.inlier_mask_
outlier_mask1 = np.logical_not(inlier_mask1)

inlier_mask2 = model_ransac2.inlier_mask_
outlier_mask2 = np.logical_not(inlier_mask2)

# Create valid line set
ransac_valid_lines = np.zeros(thetas.shape, dtype=bool)
ransac_valid_lines[setsplit] = inlier_mask1
ransac_valid_lines[~setsplit] = inlier_mask2

# Get outlier_removed set
lines = lines[ransac_valid_lines]
thetas = thetas[ransac_valid_lines]
rhos = rhos[ransac_valid_lines]

ransac_lines = lines.copy()
ransac_thetas = thetas.copy()
ransac_rhos = rhos.copy()


# print(ransac_valid_lines.shape)
# print(lines.shape)
print("Number of valid lines found:",lines.size)
print(np.floor(thetas * 180/np.pi))
print(np.floor(rhos))

# Find further largest subset of lines that are regularly spaced
# Get Sort indices
rhos1_sort_idxs = np.argsort(rhos1[inlier_mask1])
rhos1_sorted = rhos1[inlier_mask1]
rhos1_sorted = rhos1_sorted[rhos1_sort_idxs]

rhos2_sort_idxs = np.argsort(rhos2[inlier_mask2])
rhos2_sorted = rhos2[inlier_mask2]
rhos2_sorted = rhos2_sorted[rhos2_sort_idxs]

reg_spaced1 = find_regularly_spaced.getRegularlySpacedSubset(rhos1_sorted, epsilon=5)
reg_spaced2 = find_regularly_spaced.getRegularlySpacedSubset(rhos2_sorted, epsilon=5)
print("Spaced")
print(reg_spaced1)
print(reg_spaced2)

reg_space_mask1 = np.zeros(inlier_mask1.shape, dtype=bool)
reg_space_mask2 = np.zeros(inlier_mask2.shape, dtype=bool)

# Get reverse argsort indices to get valid lines of unsorted data
reverse_idx1 = np.arange(rhos1_sorted.size)
reverse_idx1[rhos1_sort_idxs] = np.arange(rhos1_sorted.size)
reverse_idx2 = np.arange(rhos2_sorted.size)
reverse_idx2[rhos2_sort_idxs] = np.arange(rhos2_sorted.size)

reg_space_mask1[reg_spaced1] = True
reg_space_mask2[reg_spaced2] = True

reg_space_mask1[reverse_idx1] = reg_space_mask1
reg_space_mask2[reverse_idx2] = reg_space_mask2

regularly_spaced_valid_lines = np.zeros(ransac_valid_lines.shape, dtype=bool)
regularly_spaced_valid_lines[setsplit] = reg_space_mask1
regularly_spaced_valid_lines[~setsplit] = reg_space_mask2
print(regularly_spaced_valid_lines)
print("Number of regularly spaced lines found:",len(reg_spaced1), len(reg_spaced2))
print(np.floor(thetas * 180/np.pi))
print(np.floor(rhos))
print("Good lines:",np.sum(regularly_spaced_valid_lines))

# Draw lines

# Draw all pruned lines
for _x1,_y1,_x2,_y2 in all_lines[:,0,:]:
  cv2.line(img,(_x1,_y1),(_x2,_y2),(250,150,150),1)

# Draw pruned lines
for _x1,_y1,_x2,_y2 in pruned_lines[:,0,:]:
  cv2.line(img,(_x1,_y1),(_x2,_y2),(155,55,250),1)

# Draw ransac outlier-removed lines
for _x1,_y1,_x2,_y2 in ransac_lines[:,0,:]:
  cv2.line(img,(_x1,_y1),(_x2,_y2),(155,155,250),2)

# Draw regularly spaced lines
for _x1,_y1,_x2,_y2 in lines[:,0,:]:
  cv2.line(img,(_x1,_y1),(_x2,_y2),(0,255,0),2)

cv2.imshow('Edges%d' % chessboard_idx,edges)
cv2.imshow('Overlay%d'%chessboard_idx,img)

# Wait for any key to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()



# plt.plot(thetas*180/np.pi,'rs')
# plt.plot((thetas*180/np.pi+90)%180,'rs')



# plt.plot(thetas*180/np.pi, rhos,'rs')
# plt.xlabel('theta')
# plt.ylabel('rho')
# plt.show()


# plt.plot(thetas1*180/np.pi, rhos1,'rs')
# plt.plot(thetas2*180/np.pi, rhos2,'bs')
# 
# 
plt.plot(X1[inlier_mask1]*180/np.pi, y1[inlier_mask1], 'gs', label='Inliers1')
plt.plot(X1[outlier_mask1]*180/np.pi, y1[outlier_mask1], 'ys', label='Outliers2')
plt.plot(X2[inlier_mask2]*180/np.pi, y2[inlier_mask2], 'go', label='Inliers2')
plt.plot(X2[outlier_mask2]*180/np.pi, y2[outlier_mask2], 'yo', label='Outliers2')
plt.plot([theta_mean*180/np.pi, theta_mean*180/np.pi], [rhos.min(), rhos.max()],'k--')
plt.show()