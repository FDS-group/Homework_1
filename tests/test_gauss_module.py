import Filtering.gauss_module as gauss_module
import numpy as np
from scipy.signal import convolve2d as conv2
from PIL import Image


def test_gauss(gaussian):
    """ Do basic tests over the gaussian function """
    x_gauss, x = gaussian

    assert len(x) == 25
    assert x_gauss[0] == x_gauss[-1]


def test_gaussian_filter():
    """ Make sure that the 1d convolution implemented provides the same result as the 2d convolution (python
    package implementation) """

    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    img = rgb2gray(np.array(Image.open('data/graf.png')))
    gx, x = gauss_module.gauss(4)
    gx = gx.reshape(1, gx.shape[0])
    gy = gx.reshape(gx.shape[1], gx.shape[0])
    smooth_img = conv2(img, gx * np.array(gy))

    test_smooth_img = gauss_module.gaussianfilter(img, 4)

    assert np.all(smooth_img.round(5) == test_smooth_img.round(5))
