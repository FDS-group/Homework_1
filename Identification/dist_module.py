import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):

    sum_of_minimum_values = np.sum(np.minimum(x, y))
    intersection_distance = 0.5*((sum_of_minimum_values/np.sum(x)) + (sum_of_minimum_values/np.sum(y)))

    assert (intersection_distance >= 0) & (intersection_distance <= 1), 'Intersection distance cannot be outside of the range [0, 1]'

    return intersection_distance

# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):

    l2_distance = np.sum((x - y)**2)

    assert (l2_distance >= 0) & (l2_distance <= np.sqrt(2)), 'Least square distance cannot be outside of the range [0, sqrt(2)]'

    return l2_distance

# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):

    # Replace zero values with small value
    x[x == 0] = 1e-10
    y[y == 0] = 1e-10

    chi_square = np.sum((x - y)**2 / (x + y))

    assert chi_square >= 0, 'Chi-square must be a positive value'

    return chi_square

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
