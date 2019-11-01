import math
import numpy as np
from functools import reduce


def symmetrize(a):
    return a + a.swapaxes(0, 1) - np.diag(a.diagonal())


def antisymmetrize(a):
    return a - a.swapaxes(0, 1) + np.diag(a.diagonal())


def closest_nice_number(number):
    oom = 10 ** math.floor(math.log10(number))
    return oom * (number // oom)


def get_linear_interpolation(p1, p2, x):
    """Given two points in 2D returns y for a given x for y = ax + b

    Parameters
    ----------
    p1,p2 : (x, y)
    x : float

    Returns
    -------
    y : float

    """
    x1, y1 = p1
    x2, y2 = p2
    a = (y2 - y1) / (x2 - x1)
    b = (x2 * y1 - x1 * y2) / (x2 - x1)
    y = a * x + b
    return y


def order_of_magnitude(number):
    """Order of magnitude of the given number

    Parameters
    ----------
    number : float

    Returns
    -------
    Float
    """
    return math.floor(math.log10(number))


def isfloat(number):
    """Check if a number or array is of float type.

    This is necessary because e.g. isinstance(np.float32(2), float) is False.

    """
    if hasattr(number, "dtype"):
        return np.issubdtype(number, np.floating)
    else:
        return isinstance(number, float)


def anyfloatin(things):
    """Check if iterable contains any non integer."""
    for n in things:
        if isfloat(n) and not n.is_integer():
            return True
    return False


def outer_nd(*vec):
    """
    Calculates outer product of n vectors

    Parameters
    ----------
    vec : vector

    Return
    ------
    out : ndarray
    """
    return reduce(np.multiply.outer, vec)


def hann_window_nth_order(m, order):
    """
    Calculates 1D Hann window of nth order

    Parameter
    ---------
    m : int
        number of points in window (typically the length of a signal)
    order : int
        Filter order

    Return
    ------
    window : array
        window
    """
    if not isinstance(m, int) or m <= 0:
        raise ValueError('Parameter m has to be positive integer greater than 0.')
    if not isinstance(order, int) or order <= 0:
        raise ValueError('Filter order has to be positive integer greater than 0.')
    sin_arg = np.pi * (m - 1.) / m
    cos_arg = 2. * np.pi / (m - 1.) * (np.arange(m))

    return m / (order * 2 * np.pi) * sum([(-1) ** i / i *
                                          np.sin(i * sin_arg) * (np.cos(i * cos_arg) - 1)
                                          for i in range(1, order + 1)])
