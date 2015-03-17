from __future__ import division
import numpy as np
from numpy import matrix
from functions import *

"""
Main functions for flow tracing.
"""


def nodeMix(F, K, P, t):
    """
    Calculate power mix of nodes.

    Inputs:
    F:  actual flows in the network
    K:  incidence matrix
    P:  injection pattern
    t:  specifying which ours of the time series to sole

    Output:
    An n by n matrix with node n's export in row n and node n's import in
    column n.
    """

    K = matrix(K)
    P = matrix(P[:, t])
    I = identity(P.shape[1])
    F = diagM(F[:, t])

    C = posM(P) * invert(I - negM(posM(K * F) * K.T))
    return C

# load test values
F = np.load('./input/F.npy')
K = np.load('./input/K.npy')
P = np.load('./input/phi.npy')
t = 0
