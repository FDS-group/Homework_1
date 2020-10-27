# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy.signal import convolve2d as conv2


"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""


def gauss(sigma):
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
    gx, x = gauss(sigma)
    gx = gx.reshape(1, gx.shape[0])
    gy = gx.reshape(gx.shape[1], gx.shape[0])
    smooth_img = scipy.signal.convolve2d(img, gx * np.array(gy))

    return smooth_img


"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""


def gaussdx(sigma):
    int_3sigma = int(sigma * 3)
    x = np.linspace(-int_3sigma, int_3sigma, (2 * int_3sigma) + 1, endpoint=True)
    dx = -(1 / (np.sqrt(2 * np.pi) * sigma ** 3)) * x * np.exp(-(x ** 2) / (2 * sigma ** 2))

    return dx, x


def gaussderiv(img, sigma):
    dx, x = gaussdx(sigma)
    dx = dx.reshape(1, dx.shape[0])
    new_row = np.zeros((((dx.shape[1] - dx.shape[0]) // 2), dx.shape[1]))
    new_matrix = np.concatenate((new_row, dx), axis=0)
    new_matrix = np.concatenate((new_matrix, new_row), axis=0)
    img= gaussianfilter(img, sigma)
    imgDx = scipy.signal.convolve2d(img, new_matrix)
    imgDy = scipy.signal.convolve2d(img, new_matrix.transpose())

    return imgDx, imgDy
