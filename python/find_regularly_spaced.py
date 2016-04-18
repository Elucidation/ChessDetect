# Given a small set of points, find largest regularly spaced subset
# with some epsilon of error allowed, brute force it all
import numpy as np

def getRegularlySpacedSubset(arr, epsilon=5):
  """Input 1D array must be sorted, return the indices of
  the largest subset of points in array that are regularly spaced"""
  N = arr.size
  best_list = []

  # For every point
  for i in range(N):
    # For every next point
    for j in range(i+1,N):
      # Get delta between them as spacing
      dx = arr[j] - arr[i]

      # Create new list of potential regularly spaced points
      spaced_list = [i,j]

      # Next predicted point given spacing dx
      nextpt = arr[j] + dx
      
      # pointer into array to check for existence of next point
      k_idx = j+1
      # Iterate over rest of array, looking for matches with spacing within 
      # a threshold of epsilon
      while nextpt < arr[-1] + epsilon and k_idx < N:
        # If point found within epsilon of predicted, add to list
        if np.abs(arr[k_idx] - nextpt) < epsilon:
          spaced_list.append(k_idx)
        nextpt = arr[k_idx] + dx
        k_idx += 1

      # If the list is larger than best so far, replace
      if len(spaced_list) > len(best_list):
        best_list = spaced_list
  # Return indices of best subset of list
  return best_list




if __name__ == '__main__':
  # Sorted list
  a = np.array([  59.5215724 ,   66.2263887 ,  103.30161888,  124.4058646 ,
                  142.05522455,  157.90806809,  179.44326234,  195.92682327,
                  216.        ,  220.4302848 ])
  print("All",a)
  best_list = getRegularlySpacedSubset(a)
  print("Best",a[best_list])