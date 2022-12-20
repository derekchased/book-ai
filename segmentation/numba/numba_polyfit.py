import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# Define Functions Using Numba
# Idea here is to solve ax = b, using least squares, where a represents our coefficients e.g. x**2, x, constants
@nb.njit
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0], deg + 1))
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x ** n
    return mat_


@nb.njit
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_


@nb.njit
def fit_poly(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


@nb.njit
def eval_polynomial(P, x):
    '''
    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0].  Uses Horner's Method.
    '''
    result = np.zeros_like(x)
    for coeff in P:
        result = x * result + coeff
    return result


if __name__ == '__main__':
    # Create Dummy Data and use existing numpy polyfit as test
    x = np.linspace(0, 2, 20)
    y = np.cos(x) + 0.3 * np.random.rand(20)

    x1 = np.array([191, 382]).astype(np.float64)
    y1 = np.array([0, 501]).astype(np.float64)

    p = np.poly1d(np.polyfit(x, y, 3))

    t = np.linspace(0, 2, 200)
    plt.plot(x, y, 'o', t, p(t), '-')

    # Now plot using the Numba (amazing) functions
    p_coeffs = fit_poly(x, y, deg=1)
    plt.plot(x, y, 'o', t, eval_polynomial(p_coeffs, t), '-')
    plt.show()
