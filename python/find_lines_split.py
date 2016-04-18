import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)


all_thetas = []
all_rhos = []

chessboard_idx = 13

for chessboard_idx in range(1,20):
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


  # Get Canny Edges
  # edges = cv2.Canny(cv2.blur(gray_img, (3,3)), 250, 256, apertureSize=3)
  edges = cv2.Canny(gray_img, 250, 256, apertureSize=3)


  # Get hough lines
  # lines = cv2.HoughLines(edges,1,np.pi/180,50)
  lines = cv2.HoughLinesP(edges,1,np.pi/180, 50, minLineLength=30, maxLineGap=50)

  if lines is None:
    print("No Lines found!")
  else:
    print("Number of lines found:",len(lines))


  thetas = []
  rhos = []

  # Draw lines and collect thetas
  if lines is not None:
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
        if np.abs(theta - other_theta) < 1.2*np.pi/180.0:
          for other_rho in rhos:
            if np.abs(rho - other_rho) < 4:
              is_duplicate = True
              break
        if is_duplicate:
          break

      if is_duplicate:
        continue


      thetas.append(theta)
      rhos.append(rho)

    print("Number of valid lines found:",len(rhos))

    # Plot all
    thetas = np.array(thetas)
    rhos = np.array(rhos)

    # Invert negative rhos
    for i in range(rhos.size):
      if thetas[i] > .75*np.pi and rhos[i] < 0:
        rhos[i] *= -1
        # thetas[i] = (thetas[i]+np.pi) % np.pi - np.pi
        thetas[i] = (thetas[i]-np.pi)

    # Add rhos/thetas to full group
    all_thetas.append(thetas)
    all_rhos.append(rhos)


# Plot all
for i in range(len(all_rhos)):
  thetas = all_thetas[i]
  rhos = all_rhos[i]
  plt.plot(thetas*180/np.pi, rhos,'s',label='%d'%i)
  theta_mean = thetas.mean()
  plt.plot([theta_mean*180/np.pi, theta_mean*180/np.pi], [rhos.min(), rhos.max()],'k--')
  plt.title('chessboard%d.jpg' % i)
  plt.show()

plt.legend()
plt.show()