import numpy as np
from distances import get_all_distances_no_param
from data.data import get_data_test


def check_identity(D, tol):
    N = D.shape[0]

    for k in range(N):
        row = D[k, :]
        col = D[:, k]

        assert np.allclose(row, col, atol=tol)


def check_diagonal(D, tol):
    assert np.allclose(np.diagonal(D), np.zeros(D.shape[0]), atol=tol)


def check_triangle(D, tol):
    N = D.shape[0]

    for k in range(N):
        row = D[k, :].reshape(1, N)
        col = D[:, k].reshape(N, 1)

        good = np.logical_or(D <= row + col, np.isclose(D, row + col, atol=tol))

        assert good.all()


# Create test data and distances
dists_no_param = get_all_distances_no_param(alphas=[0.1, 1, 10, 100, None], thresholds=[0, 0.1, 0.5, 0.7, 0.9, 1])
M = get_data_test()


# Check they are in fact distances
def test_all_distances():
    for dist in dists_no_param:
        print(dist)
        D = dist.fun(M)
        assert (D >= 0).all()
        assert (D <= 1).all()
        check_diagonal(D, tol=1e-6)
        check_triangle(D, tol=1e-6)
        check_identity(D, tol=1e-6)
