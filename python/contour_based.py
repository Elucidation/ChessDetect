import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import chessdetect_helpers
np.set_printoptions(suppress=True)


chessboard_idx = 16

# Load image
filename = 'chessboard%d.jpg' % chessboard_idx
print(filename)

original_img = cv2.imread(filename)
print("Original Shape [y, x, channels]: ",original_img.shape)

# Get scaled image
img_max_height = 320
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


im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]

contour_order = np.argsort(areas)[::-1] # Biggest to smallest


# Draw onto img as overlay
img_overlay = img.copy()
for i in range(1):
  cnt = contours[contour_order[i]]
  cv2.drawContours(img_overlay, [cnt], 0, (0,255,0), 1)

  rect = cv2.minAreaRect(cnt)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  cv2.drawContours(img_overlay, [box], 0, (0,0,255), 2)

  bbox = cv2.boundingRect(cnt)
  x,y,w,h = cv2.boundingRect(cnt)
  cv2.rectangle(img_overlay,(x,y),(x+w,y+h),(0,255,0),2)


if w < 50 or h < 50:
  print("Bbox too small")

small_img = img[y:y+h,x:x+w].copy()
small_edges = edges[y:y+h,x:x+w,]
#img, rho, theta, threshold (min num of pts in accumulator)
lines = cv2.HoughLinesP(small_edges,1,np.pi/180, 50, minLineLength=50, maxLineGap=320)
lines, thetas, rhos = chessdetect_helpers.pruneLines(lines, theta_threshold=0.5*np.pi/180.0, rho_threshold=1)

if lines is None:
  print("No Lines found!")
else:
  print("Number of lines found:",len(lines))
  for _x1,_y1,_x2,_y2 in lines[:,0,:]:
    cv2.line(small_img,(_x1,_y1),(_x2,_y2),(250,50,50),2)


cv2.imshow('Overlay',img_overlay)
cv2.imshow('Edges',edges)
cv2.imshow('Close',small_img)

# Wait for any key to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
