# If we can't find a chessboard in photo,
# try to find a bounding box for the chessboard to clip the original image before
# we resize down, and redo previous chessboard processing
import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

open_size = 3
close_size = 2
se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size,open_size))
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size,close_size))


k = 3
for chessboard_idx in range(15-k,18-k):
  # Load image
  filename = 'chessboard%d.jpg' % chessboard_idx
  print(filename)

  original_img = cv2.imread(filename)
  print("Original Shape [y, x, channels]: ",original_img.shape)

  # Get scaled image
  img_max_width = 256
  img_scale_ratio = img_max_width / original_img.shape[1]

  img = cv2.resize(original_img, (0,0), fx=img_scale_ratio, fy=img_scale_ratio)

  # Get Grayscale image
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  print("Processed Shape [y, x]: ",gray_img.shape)
  blur_img = cv2.bilateralFilter(gray_img, 10, 50,50)

  edges = cv2.cornerHarris(blur_img, 2, 3, 0.04)
  # edges = cv2.dilate(edges, None)
  edges[edges > (0.01*edges.max())] = 255



  # # Get Canny Edges
  # # edges = cv2.Canny(cv2.blur(gray_img, (3,3)),100, 256, apertureSize=3)
  # edges = cv2.Canny(blur_img,100, 256, apertureSize=3)

  edge_for_mask = edges.copy()

  mask = cv2.morphologyEx(edge_for_mask, cv2.MORPH_CLOSE, se1)
  mask2 = cv2.morphologyEx(mask.copy(), cv2.MORPH_OPEN, se2)
  # mask2 = mask2/255

  # edge_clean = edges * mask2

  cv2.imshow('Image%d'%chessboard_idx,blur_img)
  cv2.imshow('Edges%d' % chessboard_idx,edges)
  cv2.imshow('EdgesClean%d' % chessboard_idx, mask2)
  cv2.waitKey(10)


  # Wait for any key to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()