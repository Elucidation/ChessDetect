import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

chessboard_idx = 16

# Load image
filename = 'chessboard%d.jpg' % chessboard_idx
print("File:",filename)

original_img = cv2.imread(filename)
print("Original Shape [y, x, channels]: ",original_img.shape)

# Get scaled image
img_max_height = 320.
img_scale_ratio = img_max_height / original_img.shape[1]


img = cv2.resize(original_img, (0,0), fx=img_scale_ratio, fy=img_scale_ratio)
img_width = img.shape[0]

# Get Grayscale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.blur(gray_img, (3,3))
gray_img = cv2.bilateralFilter(gray_img, 10, 25,25) # Better but slower, only use for final thing

# Get Canny Edges
# edges = cv2.Canny(cv2.blur(gray_img, (3,3)), 250, 256, apertureSize=3)
edges = cv2.Canny(gray_img, 250, 256, apertureSize=3)

close_size = 10
se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size,close_size))

mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, se1)


print("Processed Shape [y, x]: ",gray_img.shape)

corners = cv2.cornerHarris(edges, 2, 3, 0.04)

# cv2.imshow('Overlay',img_overlay)
cv2.imshow('Edges',edges)
cv2.imshow('Corners',(corners>0)*255)

# Wait for any key to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
