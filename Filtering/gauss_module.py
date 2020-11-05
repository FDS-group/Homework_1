# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
from scipy.signal import convolve2d as conv2

"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""


def gauss(sigma):
    """
    Gauss Function

    :param sigma: (float) Sigma value of Gaussian
    :return: Gaussian array and x array
    """
    int_3sigma = int(sigma * 3)
    x = np.linspace(-int_3sigma, int_3sigma, (2 * int_3sigma) + 1, endpoint=True)
    gx = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x ** 2) / (2 * sigma ** 2))

    return gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""


def gaussianfilter(img, sigma):
    """
    Apply gaussian filter over image

    :param img: (np.array) Image matrix over which filter will be applied
    :param sigma: (float) Sigma value of Gaussian
    :return: Filtered image
    """

    gx, x = gauss(sigma)
    gx = gx.reshape(1, gx.shape[0])
    gy = gx.reshape(gx.shape[1], gx.shape[0])
    smooth_img_x = conv2(img, gx)
    smooth_img = conv2(smooth_img_x, gy)

    return smooth_img


"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""


def gaussdx(sigma, cap=False):
    int_3sigma = int(sigma * 3)
    x = np.linspace(-int_3sigma, int_3sigma, (2 * int_3sigma) + 1, endpoint=True)
    if cap:
        x[x < -cap] = -cap
        x[x > cap] = cap
    dx = -(1 / (np.sqrt(2 * np.pi) * sigma ** 3)) * x * np.exp(-(x ** 2) / (2 * sigma ** 2))

    return dx, x


def gaussderiv(img, sigma, cap=False, smoothen_image=True):
    dx, x = gaussdx(sigma, cap)
    dx = dx.reshape(1, dx.shape[0])
    dy = dx.reshape(dx.shape[1], dx.shape[0])
    if smoothen_image:
        img = gaussianfilter(img, sigma)
    imgDx = conv2(img, dx, mode='same')
    imgDy = conv2(img, np.array(dy), mode='same')
    return imgDx, imgDy
