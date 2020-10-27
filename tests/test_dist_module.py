import Identification.dist_module as dist_module
import numpy as np


def test_dist_l2_ranges():
    """ Test function distance l2 """

    x = np.random.uniform(size=100)
    y = x
    distance = dist_module.dist_l2(x, y)

    assert distance == 0


def test_dist_chi2_ranges():
    """ Test function distance chi2 """

    x = np.random.uniform(size=100)
    y = x
    distance = dist_module.dist_chi2(x, y)

    assert distance == 0
