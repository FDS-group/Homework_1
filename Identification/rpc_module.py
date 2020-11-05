import numpy as np
import matplotlib.pyplot as plt

import Identification.match_module as match_module


# TP = number of correct matches among images with distance smaller then tau
# FP = number of incorrect matches among images with distance smaller then tau
# FN = number of correct matches among images with distance larger then tau


# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between query image i, and model image j
# i.e.:
#        (distance_query1_model_1, distance_query1_model2, ....., distance_query1_modeln)
# D =    (distance_query2_model_1, distance_query2_model2, ....., distance_query2_modeln)
#                                              ....
#        (distance_queryn_model_1, distance_queryn_model2, ....., distance_queryn_modeln)

# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th
# model image

def plot_rpc(D, plot_color):
    recall = []
    precision = []
    num_queries = D.shape[1]

    num_images = D.shape[0]
    assert (num_images == num_queries), 'Distance matrix should be a square matrix'

    labels = np.diag([1] * num_images)

    # Convert D and l into a 1d array
    d = D.reshape(D.size)
    l = labels.reshape(labels.size)

    sortidx = d.argsort()
    d = d[sortidx]
    l = l[sortidx]

    tp = 0

    # Loop over all distance values. These distance values will also provide us the different tau distances which we
    # will be increasing step by step
    for idt in range(len(d)):
        tp += l[idt]
        fp = len(l[:idt]) + 1 - tp
        fn = np.sum(l[idt + 1:])

        # Compute precision and recall values and append them to "recall" and "precision" vectors
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))

    plt.plot([1 - precision[i] for i in range(len(precision))], recall, plot_color + '-')


def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):
    assert len(plot_colors) == len(dist_types), 'number of distance types should match the requested plot colors'

    for idx in range(len(dist_types)):
        [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)
        plot_rpc(D, plot_colors[idx])

    plt.axis([0, 1, 0, 1])
    plt.xlabel('1 - precision')
    plt.ylabel('recall')

    plt.legend(dist_types, loc='best')
