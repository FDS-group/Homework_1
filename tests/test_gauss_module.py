

def test_gauss(gaussian):
    """ Do basic tests over the gaussian function """
    x_gauss, x = gaussian

    assert len(x) == 25
    assert x_gauss[0] == x_gauss[-1]
