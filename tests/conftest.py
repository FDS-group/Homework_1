from pytest import fixture
import numpy as np

import Filtering.gauss_module as gauss_module

# ---------------- Gauss module ----------------


@fixture
def gaussian():
    sigma = 4
    return gauss_module.gauss(sigma)


@fixture
def construct_dummy_images():
    width = 126

    red_image = np.zeros((width, width, 3), np.uint8)
    green_image = np.zeros((width, width, 3), np.uint8)
    blue_image = np.zeros((width, width, 3), np.uint8)
    mix_image = np.zeros((width, width, 3), np.uint8)

    red_image[:, :] = (255, 0, 0)
    green_image[:, :] = (0, 255, 0)
    blue_image[:, :] = (0, 0, 255)
    mix_image[:, 0:width // 3] = (255, 0, 0)
    mix_image[:, width // 3:(width // 3) * 2] = (0, 255, 0)
    mix_image[:, (width // 3) * 2:width] = (0, 0, 255)

    return red_image, green_image, blue_image, mix_image
