import numpy as np
import Identification.histogram_module as hist_module


def test_rgb_hist(construct_dummy_images):
    """ test rgb hist over uniform images """

    red_image, green_image, blue_image, mix_image = construct_dummy_images

    num_bins = 5
    red_hist = hist_module.rgb_hist(red_image.astype('double'), num_bins)
    green_hist = hist_module.rgb_hist(green_image.astype('double'), num_bins)
    blue_hist = hist_module.rgb_hist(blue_image.astype('double'), num_bins)
    mix_hist = hist_module.rgb_hist(mix_image.astype('double'), num_bins)

    assert len(red_hist[red_hist == 1]) == 1
    assert len(green_hist[green_hist == 1]) == 1
    assert len(blue_hist[blue_hist == 1]) == 1
    assert len(mix_hist[mix_hist != 0]) == 3


def test_rg_hist(construct_dummy_images):
    """ test rgb hist over uniform images """

    red_image, green_image, blue_image, mix_image = construct_dummy_images

    num_bins = 5
    red_hist = hist_module.rg_hist(red_image.astype('double'), num_bins)
    green_hist = hist_module.rg_hist(green_image.astype('double'), num_bins)
    blue_hist = hist_module.rg_hist(blue_image.astype('double'), num_bins)
    mix_hist = hist_module.rg_hist(mix_image.astype('double'), num_bins)

    assert len(red_hist[red_hist == 1]) == 1
    assert len(green_hist[green_hist == 1]) == 1
    assert len(blue_hist[blue_hist == 1]) == 1
    assert len(mix_hist[mix_hist != 0]) == 3
