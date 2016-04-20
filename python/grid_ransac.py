import cv2
import numpy as np
from matplotlib import pyplot as plt
import chessdetect_helpers
np.set_printoptions(suppress=True)

# Ransac grid:
# Iterate up to k times:
#   1. Select 4 random points as hypothetical inlier (skip 4 points too close together, always order bottom-left top-left... based on cartesian coords to simplify)
#   2. Find homography M to the unit square (0,0 0,1 1,1 1,0 corners)
#   3. Warp all features with M, 
#     measure distance of each point to its closest round number grid coordinate
#     threshold based on distance to determine consensus set
#   4. Replace best consensus count, if consensus set size > threshold, break
# 5. After iteration, recalculate homography using all members of consensus set (calculate closest grid points from previous 4 points and use that in homography)


chessboard_idx = 12

# Load image
filename = 'chessboard%d.jpg' % chessboard_idx
print(filename)

original_img = cv2.imread(filename)
print("Original Shape [y, x, channels]: ",original_img.shape)

# Get scaled image
img_max_height = 512.
img_scale_ratio = img_max_height / original_img.shape[1]


img = cv2.resize(original_img, (0,0), fx=img_scale_ratio, fy=img_scale_ratio)
img_width = img.shape[0]

# Get Grayscale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.blur(gray_img, (3,3))
# gray_img = cv2.bilateralFilter(gray_img, 10, 25,25) # Better but slower, only use for final thing

# Does pretty much all the stuff we need
corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.01, img_max_height/50.0)

######################
# RANSAC!
best_idxs = None
best_consensus = np.array([])
best_M1 = None
for k in range(100):
  # 1
  # start_idxs = np.array([34,33,18,43]) # Manually chosen, ransac will guess something like this later
  for o in range(100):
    start_idxs = np.random.choice(corners.shape[0], 4, replace=False)
    srcpts = corners[start_idxs,0,:]
    # Skip bad set of points which are too close to each other
    bad_set = False
    for i in range(4):
      for j in range(i+1,4):
        d = np.linalg.norm(srcpts[i,:] - srcpts[j,:])
        if d < 15 or d > 100:
          bad_set = True
          break
    if bad_set:
      # print "Bad set"
      continue
    else:
      break

  dstpts = np.array([[0,0],[0,1],[1,1],[1,0]],dtype=np.float32) # unit square

  # 2 Initial homography 4 pts to 4 pts
  M1, _ = cv2.findHomography(srcpts, dstpts)
  # print(M1)

  # 3 Warp all pts
  warpedpts = (np.mat(M1) * np.hstack([corners[:,0,:], np.ones((corners.shape[0],1))]).T).T
  warpedpts /= warpedpts[:,2] # normalize by z

  # Find consensus
  consensus = np.sum((warpedpts[:,:2] - warpedpts[:,:2].round()).A **2, 1) < 0.01
  if (consensus.sum() > best_consensus.sum()):
    best_consensus = consensus.copy()
    best_M1 = M1.copy()
    best_idxs = start_idxs.copy()
  print(k, " | consensus", consensus.sum(), "vs best", best_consensus.sum())
  if (consensus.sum() > 49):
    break


consensus = best_consensus.copy()

# 3 Warp all pts with best consensus
warpedpts = (np.mat(best_M1) * np.hstack([corners[:,0,:], np.ones((corners.shape[0],1))]).T).T
warpedpts /= warpedpts[:,2] # normalize by z

# 5 use all members in consensus to recalculate closest grid points
# Refine a few times to keep adding more points
for i in range(10):
  srcpts = corners[consensus,0,:]
  dstpts = warpedpts[consensus,:2].round().astype(np.float32)
  M2, _ = cv2.findHomography(srcpts, dstpts)

  warpedpts = (np.mat(M2) * np.hstack([corners[:,0,:], np.ones((corners.shape[0],1))]).T).T
  warpedpts /= warpedpts[:,2] # normalize by z

  # Find consensus
  consensus = np.sum((warpedpts[:,:2] - warpedpts[:,:2].round()).A **2, 1) < 0.01
  # print(i,"| iterate new consensus", consensus.sum())
print(M2)


#####################



### CV draw overlays
# Overlay corners onto image
img_overlay = img.copy()
for pt in corners[~consensus,0,:]:
  cv2.circle(img_overlay, tuple(pt), 1, (0,0,255),2)
for pt in corners[consensus,0,:]:
  cv2.circle(img_overlay, tuple(pt), 2, (0,255,0),3)

for pt in corners[best_idxs,0,:]:
  cv2.circle(img_overlay, tuple(pt), 4, (255,255,0),5)

cv2.imshow('Overlay',img_overlay)
cv2.moveWindow('Overlay', 0,0)
# cv2.imshow('Edges',gray_img)

# Wait for any key to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# # plt.plot(corners[:,0,0], corners[:,0,1], 'ro')
# plt.plot(warpedpts[consensus,0], warpedpts[consensus,1], 'go')
# plt.plot(warpedpts[~consensus,0], warpedpts[~consensus,1], 'ro')
# plt.axis('equal')
# plt.grid()
# plt.show()