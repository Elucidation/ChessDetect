import numpy as np

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