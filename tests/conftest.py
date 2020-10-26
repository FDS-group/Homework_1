from pytest import fixture

import Filtering.gauss_module as gauss_module

# ---------------- Gauss module ----------------


@fixture
def gaussian():
    sigma = 4
    return gauss_module.gauss(sigma)
