import numpy as np
from numpy import histogram as hist
import Filtering.gauss_module as gauss_module

# Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)


#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
# TODO: rethink algorithm
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # Compute bin values
    bins = np.linspace(0, 255, num_bins + 1)
    # Unnest array and sort
    img_gray = np.sort(img_gray.reshape(img_gray.size))
    # Set empty list to complete histogram
    hists = [0] * num_bins
    # Loop over all image pixel values assigning the values to the corresponding bins
    position = 1
    for i in range(len(img_gray)):
        if img_gray[i] <= bins[position]:
            hists[position - 1] += 1
        else:
            while img_gray[i] >= bins[position]:
                position += 1
            hists[position - 1] += 1
    # Normalize
    hists = np.array(hists) / sum(hists)

    assert np.sum(hists) == 1, 'Histogram is not normalized'

    return hists, bins


#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins, reshape='C'):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    # Convert color image into a 2d array
    img_color_double_reshaped = img_color_double.reshape(-1, 3)

    # Bins (these will apply over the three dimensions equally)
    bins = np.linspace(0, 255 + 1e-10, num_bins + 1)

    # Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))

    # Loop for each pixel i in the image
    for i in img_color_double_reshaped.tolist():
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        # Identify where the value of pixel i would fall into
        indexR = int(np.digitize(i[0], bins, right=False)) - 1
        indexG = int(np.digitize(i[1], bins, right=False)) - 1
        indexB = int(np.digitize(i[2], bins, right=False)) - 1

        hists[indexR, indexG, indexB] += 1

    # Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / np.sum(hists)

    # Return the histogram as a 1D vector
    hists = np.reshape(hists, num_bins**3, order=reshape)

    assert np.sum(hists) == 1, 'Histogram is not normalized'

    return hists


#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    # Convert color image into a 2d array
    img_color_double_reshaped = img_color_double.reshape(-1, 3)

    # Bins (these will apply over the three dimensions equally)
    bins = np.linspace(0, 255 + 1e-10, num_bins + 1)

    # Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    # Loop for each pixel i in the image
    for i in img_color_double_reshaped.tolist():
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        # Identify where the value of pixel i would fall into
        indexR = int(np.digitize(i[0], bins, right=False)) - 1
        indexG = int(np.digitize(i[1], bins, right=False)) - 1

        hists[indexR, indexG] += 1

    # Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / np.sum(hists)

    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    assert np.sum(hists) == 1, 'Histogram is not normalized'

    return hists


#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    imgDx, imgDy = gauss_module.gaussderiv(img_gray, 3)

    imgDx[imgDx < -6] = -6
    imgDx[imgDx > 6] = 6
    imgDy[imgDy < -6] = -6
    imgDy[imgDy > 6] = 6

    # Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    bins = np.linspace(min(np.min(imgDx), np.min(imgDy)), max(np.max(imgDx), np.max(imgDy)) + 1e-10, num_bins + 1)

    # Resize imgDx and imgDy to do the loop in a more readable manner
    imgDx = imgDx.reshape(imgDx.size)
    imgDy = imgDy.reshape(imgDy.size)

    assert len(imgDx) == len(imgDy), 'imgDx and imgDy should have the same dimension!'

    for i in range(len(imgDx)):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        # Identify where the value of pixel i would fall into
        index_dx = int(np.digitize(imgDx[i], bins, right=False)) - 1
        index_dy = int(np.digitize(imgDy[i], bins, right=False)) - 1

        hists[index_dx, index_dy] += 1

    # Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / np.sum(hists)

    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists


def is_grayvalue_hist(hist_name):
    if hist_name == 'grayvalue' or hist_name == 'dxdy':
        return True
    elif hist_name == 'rgb' or hist_name == 'rg':
        return False
    else:
        assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
    if hist_name == 'grayvalue':
        return normalized_hist(img, num_bins_gray)
    elif hist_name == 'rgb':
        return rgb_hist(img, num_bins_gray)
    elif hist_name == 'rg':
        return rg_hist(img, num_bins_gray)
    elif hist_name == 'dxdy':
        return dxdy_hist(img, num_bins_gray)
    else:
        assert False, 'unknown distance: %s' % hist_name
