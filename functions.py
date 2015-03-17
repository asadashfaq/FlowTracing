from __future__ import division
import numpy as np
from numpy import matrix

"""
A collection of commonly used functions.
"""


def isMatrix(M):
    """
    Check if input is matrix, if not convert to matrix.
    """
    if type(M) == matrix:
        return M
    elif type(M) == np.ndarray:
        return matrix(M)
    else:
        raise Exception('Unknown input format. Should be matrix or numpy array')


def identity(n):
    """
    Create an identity matrix.
    """
    I = np.zeros((n, n))
    diag = np.ones(n)
    np.fill_diagonal(I, diag)
    return matrix(I)


def invert(M):
    """
    Invert matrix if possible.
    Need check of inverse and possibly implementation of pseudo inverse.
    """
    M = isMatrix(M)
    return M.I


def diagM(l):
    """
    Return input list as diagonal matrix.
    """
    dim = len(l)
    M = np.zeros((dim, dim))
    np.fill_diagonal(M, l)
    return matrix(M)


def posM(M):
    """
    Return matrix with negative values set to zero.
    """
    M[np.where(M < 0)] = 0
    return M


def negM(M):
    """
    Return matrix with negative values set to zero.
    """
    M[np.where(M > 0)] = 0
    return M
