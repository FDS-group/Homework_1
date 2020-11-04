import Identification.dist_module as dist_module
import numpy as np


def test_dist_intersection_ranges():
    """ Test function distance l2 """

    x = np.random.uniform(size=100)
    x = x/np.sum(x)
    y = x
    distance = dist_module.dist_intersect(x, y)

    assert round(distance, 6) == 0


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


def test_dist_intersect():
    """ Test function distance chi2 """

    A = np.array(
        [1.274580708, 2.466224824, 5.045757621, 7.413716262, 8.958855646, 10.41325305, 11.14150951, 10.91949012,
         11.29095648, 10.95054297, 10.10976255, 8.128781795, 1.886568472])
    B = np.array(
        [0, 1.700493692, 4.059243006, 5.320899616, 6.747120132, 7.899067471, 9.434997257, 11.24520022, 12.94569391,
         12.83598464, 12.6165661, 10.80636314, 4.388370817])

    distance = 1 - dist_module.dist_intersect(A/np.sum(A), B/np.sum(B))

    assert round(distance, 6) == round(88.447923561/100, 6)


def test_dist_chi2():
    """ Test function distance chi2 """

    A = np.array([1, 2, 13, 5, 45, 23])
    B = np.array([67, 90, 18, 79, 24, 98])

    distance = dist_module.dist_chi2(A, B)

    assert round(distance, 6) == 2*round(133.55428601494035, 6)



