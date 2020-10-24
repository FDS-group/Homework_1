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
def gauss(sigma):
    """
    Gauss Function
    :param sigma: (float) Sigma value of Gaussian
    :param steps: (int) Steps of the x array
    :return: Gaussian array
    """

    x = np.arange(-3*sigma, 3*sigma+1e-10, 1)
    gx = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x**2)/(2*sigma**2))

    return gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
# TODO: (nice to have) implement convolution ourselves
def gaussianfilter(img, sigma):

    gauss_kernel_1d, _ = gauss(sigma)
    gauss_kernel = np.dot(gauss_kernel_1d[:, None], gauss_kernel_1d[None, :])

    smooth_img = conv2(img, gauss_kernel)
    smooth_img_normal = (255*(smooth_img-np.min(smooth_img)))/(np.max(smooth_img)-np.min(smooth_img))


    return smooth_img_normal



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    #...
    
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

