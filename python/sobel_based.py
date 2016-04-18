import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
np.set_printoptions(suppress=True)


chessboard_idx = 4

# Load image
filename = 'chessboard%d.jpg' % chessboard_idx
print(filename)

original_img = cv2.imread(filename)
print("Original Shape [y, x, channels]: ",original_img.shape)

# Get scaled image
img_max_height = 256
img_scale_ratio = img_max_height / original_img.shape[1]


img = cv2.resize(original_img, (0,0), fx=img_scale_ratio, fy=img_scale_ratio)
img_width = img.shape[0]

# Get Grayscale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.blur(gray_img, (3,3))
gray_img = cv2.bilateralFilter(gray_img, 10, 25,25)

print("Processed Shape [y, x]: ",gray_img.shape)


sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1,0, ksize=5)
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0,1, ksize=5)

# Range -1 to 1
sobelx_normalized = ( (sobelx - sobelx.min())/(sobelx.max()-sobelx.min()) - 0.5) * 2.0
sobely_normalized = ( (sobely - sobely.min())/(sobely.max()-sobely.min()) - 0.5) * 2.0

# Binary Grayscale 0 and 255
sobelx_thresholded = np.array(np.abs(sobelx_normalized)>0.2, dtype=np.uint8) * 255
sobely_thresholded = np.array(np.abs(sobely_normalized)>0.2, dtype=np.uint8) * 255

# Phase
# phase = cv2.phase(sobelx, sobely, angleInDegrees=False)

# Find bounding box of gradients where there are at least 
# 1/4 of the image number of pixels of whole image in the columns and rows respectively
sobelx_limit_mask = sobelx_thresholded.sum(0)/255 > img_max_height/8
sobely_limit_mask = sobely_thresholded.sum(1)/255 > img_width/8

if not np.any(sobelx_limit_mask) or not np.any(sobely_limit_mask):
  raise("No gradients passed threshold")

sobelx_limits = np.nonzero(sobelx_limit_mask)[0]
sobely_limits = np.nonzero(sobely_limit_mask)[0]
bbox_x1, bbox_x2 = sobelx_limits[0], sobelx_limits[-1]
bbox_y1, bbox_y2 = sobely_limits[0], sobely_limits[-1]

print((bbox_x1,bbox_y1),(bbox_x2,bbox_y2))

cv2.rectangle(img, (bbox_x1,bbox_y1), (bbox_x2,bbox_y2), (0,0,255),2)

cv2.imshow('Overlay',img)
cv2.imshow('Gray',gray_img)
cv2.imshow('SobelX',sobelx_thresholded)
cv2.imshow('SobelY',sobely_thresholded)

# Wait for any key to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sum of equally
plt.plot(sobelx_limit_mask)
plt.plot(sobely_limit_mask)
plt.show()






# phase_vals = phase_thresholded.flatten() * 180/np.pi

# plt.plot(phase_vals, ',')
# plt.show()