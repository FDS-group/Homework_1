import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import Identification.histogram_module as histogram_module
import Identification.dist_module as dist_module


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins, nearest_neighbours=None):
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

    model_images = ['Identification/' + model_images[i] for i in range(len(model_images))]
    query_images = ['Identification/' + query_images[i] for i in range(len(query_images))]
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    D = np.zeros((len(query_images), len(model_images)))

    for i in range(len(query_images)):
        for j in range(len(model_images)):
            D[i, j] = dist_module.get_dist_by_name(query_hists[i], model_hists[j], dist_type)

    if nearest_neighbours:
        sorted_D = np.argsort(D, axis=1)[:, :nearest_neighbours]
        return sorted_D, D
    else:
        best_match = np.argmin(D, axis=1)
        return best_match, D


# TODO: use hist_isgray to generate the dxdy hist
def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    image_hist = []

    # Compute histogram for each image and add it at the bottom of image_hist
    if hist_isgray:
        for image in image_list:
            img_color = np.array(Image.open(image)).astype('double')
            image_matrix = rgb2gray(img_color)
            image_hist.append(histogram_module.get_hist_by_name(img=image_matrix, num_bins_gray=num_bins,
                                                                hist_name=hist_type))
    else:
        for image in image_list:
            image_matrix = np.array(Image.open(image)).astype('double')
            image_hist.append(histogram_module.get_hist_by_name(img=image_matrix, num_bins_gray=num_bins,
                                                                hist_name=hist_type))

    return image_hist


# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image


def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    neighbors_index, _ = find_best_match(model_images, query_images, dist_type, hist_type, num_bins,
                                         nearest_neighbours=num_nearest)
    fig = plt.figure(figsize=(12, 8))
    columns = num_nearest + 1
    rows = len(query_images)
    for i in range(1, columns * rows + 1):
        if i % columns == 1:
            img = np.array(Image.open('Identification/' + query_images[i // columns]))
            fig.add_subplot(rows, columns, i)
            plt.title(query_images[i // columns])
            plt.imshow(img)
        else:
            image = np.array(
                Image.open('Identification/' + model_images[neighbors_index[(i - 1) // columns, (i - 2) % columns]]))
            fig.add_subplot(rows, columns, i)
            plt.title(model_images[neighbors_index[(i - 1) // columns, (i - 2) % columns]])
            plt.imshow(image)
    fig.tight_layout(pad=2.0)
    plt.show()
