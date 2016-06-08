import cv2
import numpy as np
import chessdetect_helpers as hf
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)


for chessboard_idx in range(19,20):

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

  # Get lines
  lines, thetas, rhos = hf.getLines(gray_img)
  print(rhos)

  if lines is None:
    print("No Lines found!")
    continue
  else:
    print("Number of lines found:",len(lines))

  overlay_img = img.copy()
  for x1,y1,x2,y2 in lines:
    cv2.line(overlay_img,(x1,y1),(x2,y2),(0,255,255),1)

  # Plot hough space lines as points
  plt.plot(thetas*180/np.pi, rhos,'s',label='%d'%chessboard_idx)
  theta_mean = thetas.mean()
  plt.plot([theta_mean*180/np.pi, theta_mean*180/np.pi], [rhos.min(), rhos.max()],'k--')
  plt.title('chessboard%d.jpg' % (chessboard_idx))

  # Show overlaid image, and show plot
  cv2.imshow('Overlay%d'%chessboard_idx,overlay_img)
  cv2.waitKey(10)
  plt.show()
  cv2.destroyAllWindows()