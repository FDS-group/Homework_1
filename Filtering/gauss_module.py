# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2

"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""


# TODO: test function
# TODO: why is the sum of the gaussian not exactly equal to 1?
def gauss(sigma):
    """
    Gauss Function

    :param sigma: (float) Sigma value of Gaussian
    :return: Gaussian array and x array
    """
    sigma = int(sigma)
    x = np.linspace(-3 * sigma, 3 * sigma, (2 * 3 * sigma) + 1, endpoint=True)
    gx = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x ** 2) / (2 * sigma ** 2))

    return gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""


# TODO: Remove convolution_1d input parameter once we are sure they both return same result (first test return same
#  result)
# TODO: 1d convolution seems to go slower than 2d convolution...
def gaussianfilter(img, sigma, convolution_1d=True):
    gauss_kernel_1d, _ = gauss(sigma)
    len_1d_kernel = len(gauss_kernel_1d)

    if not convolution_1d:
        gauss_kernel = np.dot(gauss_kernel_1d[:, None], gauss_kernel_1d[None, :])

        smooth_img = conv2(img, gauss_kernel)
    else:
        # Perform convolution using 1d kernel filters
        x_axis_convolution = []
        for i in range(img.shape[0]):
            x_axis_convolution.append(
                conv2(img[i].reshape((1, img.shape[1])), gauss_kernel_1d.reshape((1, len_1d_kernel)))[0].tolist())
        y_axis_convolution = []
        for i in range(np.array(x_axis_convolution).shape[1]):
            y_axis_convolution.append(
                conv2(np.array(x_axis_convolution)[:, i].reshape((1, np.array(x_axis_convolution).shape[0])),
                      gauss_kernel_1d.reshape((1, len_1d_kernel)))[0].tolist())
        smooth_img = np.array(y_axis_convolution).T

    return smooth_img


"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""


# TODO: test
def gaussdx(sigma):

    sigma = int(sigma)
    x = np.arange(-3 * sigma, 3 * sigma + 1e-10, 1)
    dx = -(1 / (np.sqrt(2 * np.pi) * sigma ** 3)) * x * np.exp(-(x ** 2) / (2 * sigma ** 2))

    return dx, x


def gaussderiv(img, sigma, smoothen_image_first=True):
    """ Use 1d kernel gaussdx (for x and y) and apply over the img """

    if smoothen_image_first:
        img = gaussianfilter(img, sigma)

    gaussdx_kernel_1d, _ = gaussdx(sigma)
    len_1d_kernel = len(gaussdx_kernel_1d)

    # Perform convolution using 1d kernel filters
    imgdx = []
    for i in range(img.shape[0]):
        imgdx.append(
            conv2(img[i].reshape((1, img.shape[1])), gaussdx_kernel_1d.reshape((1, len_1d_kernel)))[0].tolist())
    imgdy = []
    for i in range(img.shape[1]):
        imgdy.append(
            conv2(img[:, i].reshape((1, img.shape[0])), gaussdx_kernel_1d.reshape((1, len_1d_kernel)))[0].tolist())
    imgdy = np.array(imgdy).T

    return imgdx, imgdy
