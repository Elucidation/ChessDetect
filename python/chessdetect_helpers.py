import numpy as np
import cv2

def pruneLines(lines, theta_threshold=1.2*np.pi/180.0, rho_threshold=6):
  thetas = []
  rhos = []

  # Draw lines and collect thetas
  good_lines = []
  if lines is not None:
    for x1,y1,x2,y2 in lines[0,:,:]:
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
      # for other_theta in thetas:
      #   if np.abs(theta - other_theta) < theta_threshold:
      #     for other_rho in rhos:
      #       if np.abs(rho - other_rho) < rho_threshold:
      #         is_duplicate = True
      #         break
      #   if is_duplicate:
      #     break

      # Only add non-duplicate thetas/rhos
      if is_duplicate:
        good_lines.append(False)
      else:
        thetas.append(theta)
        rhos.append(rho)
        good_lines.append(True)
  
  return [lines[:,np.array(good_lines,dtype=bool),:], thetas, rhos]

def getLines(img):
  """Return houghlinesP + thetas and rhos for each lines from given image"""
  if img.ndim == 3:
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Get Canny Edges
  # edges = cv2.Canny(cv2.blur(gray_img, (3,3)), 250, 256, apertureSize=3)
  edges = cv2.Canny(img, 250, 256, apertureSize=3)

  lines = cv2.HoughLinesP(edges,1,np.pi/180, 50, minLineLength=60, maxLineGap=50)

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

      # Invert negative rho
      if theta > 0.75*np.pi and rho < 0:
        rho *= -1
        theta -= np.pi

      thetas.append(theta)
      rhos.append(rho)

  thetas = np.array(thetas)
  rhos = np.array(rhos)

  return (lines[:,0,:],thetas,rhos)
