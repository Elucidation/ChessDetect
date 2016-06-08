import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

chessboard_idx = 2

# Load image
filename = 'chessboard%d.jpg' % chessboard_idx
print(filename)

original_img = cv2.imread(filename)
print("Original Size [y, x, channels]: ",original_img.shape)

# Get scaled image
img_max_height = 320.
img_scale_ratio = img_max_height / original_img.shape[1]


img = cv2.resize(original_img, (0,0), fx=img_scale_ratio, fy=img_scale_ratio)
img_width = img.shape[0]

# Get Grayscale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Size for processing [y, x]: ",gray_img.shape)

# Smooth it out
# gray_img = cv2.blur(gray_img, (5,5))
# gray_img = cv2.bilateralFilter(gray_img, 10, 25,25)

### Image Chessboard points
# Corners
corners = cv2.goodFeaturesToTrack(gray_img, 49, 0.01, 15)

# Generate keypoints
image_pts = []
# for x,y in corners[:,0,:]:
  # image_pts.append(cv2.Keypoint(x,y,1))

### TEMPLATE Chessboard points
# Set of chessboard corners 7x7 to fit a homography to
x = np.tile(np.arange(7, dtype=np.float32),(7,1))
chessboard_pts = np.hstack([x.reshape(-1,1),x.reshape(-1,1,order='F')]).reshape(-1,1,2)*20

# Generate keypoints for template

# Find homography between chessboard corners to image features
homography, mask = cv2.findHomography(chessboard_pts, corners, method=cv2.RANSAC, ransacReprojThreshold=5)
print(homography)


# Draw corners as circles onto view
img_corners = img.copy()
for x,y in corners[mask.flatten() > 0,0,:]:
  cv2.circle(img_corners, (x,y), 2, (0,255,0), 2)

for x,y in corners[mask.flatten() == 0,0,:]:
  cv2.circle(img_corners, (x,y), 1, (0,0,255), 2)

cv2.imshow('Overlay',img)
cv2.imshow('Gray',gray_img)
cv2.imshow('Corners',img_corners)

# Wait for any key to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()