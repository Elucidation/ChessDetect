import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)


for chessboard_idx in range(10,15):
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
  edges = cv2.Canny(cv2.blur(gray_img, (3,3)), 250, 256, apertureSize=3)
  # edges = cv2.Canny(gray_img, 250, 256, apertureSize=3)


  # Get hough lines
  # lines = cv2.HoughLines(edges,1,np.pi/180,50)
  lines = cv2.HoughLinesP(edges,1,np.pi/180, 50, minLineLength=80, maxLineGap=50)

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

      # Plot
      cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

  thetas = np.array(thetas)
  rhos = np.array(rhos)
  print("Number of valid lines found:",len(rhos))
  print(np.floor(thetas * 180/np.pi))
  print(np.floor(rhos))
  # plt.plot(thetas*180/np.pi,rhos,'rs')

  # cv2.imshow('Color',img)
  # cv2.moveWindow('Color',0,0)
  # cv2.imshow('Gray', gray_img)
  # cv2.moveWindow('Gray',gray_img.shape[1]+2,0)
  cv2.imshow('Edges%d' % chessboard_idx,edges)
  # cv2.moveWindow('Edges',0,gray_img.shape[0]+32)
  cv2.imshow('Overlay%d'%chessboard_idx,img)
  cv2.waitKey(10)
  # cv2.moveWindow('Overlay',gray_img.shape[1]+2,gray_img.shape[0]+32)


  # Wait for any key to destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

  # plt.plot(thetas*180/np.pi,'rs')
  # plt.show()