from __future__ import division
import numpy as np
from numpy import matrix
from functions import *

"""
Main functions for flow tracing.
"""


def nodeMix(F, K, P, t, dir):
    """
    Calculate power mix of nodes.

    Inputs:
    F:      actual flows in the network
    K:      incidence matrix
    P:      injection pattern
    t:      specifying which ours of the time series to sole either an integer
            or a list of time steps.
    dir:    'export' or 'import'

    Output:
    An n by n matrix with node n's export in row n and node n's import in
    column n.
    """
    K = isMatrix(K)
    dim = K.shape[0]
    I = identity(dim)

    if type(t) == int:
        timeSteps = [t]
        C = np.zeros((1, dim, dim))
    elif type(t) == list:
        timeSteps = t
        C = np.zeros((len(t), dim, dim))

    if dir == 'export':
        for i, t in enumerate(timeSteps):
            P = diagM(P[:, t])
            F = diagM(F[:, t])
            C[i] = posM(P) * invert(I - negM(posM(K * F) * K.T))
    elif dir == 'import':
        for i, t in enumerate(timeSteps):
            P = diagM(P[:, t])
            F = diagM(F[:, t])
            C[i] = abs(negM(P)) * invert(I + negM(K * F) * K.T)
    return C


def linkMix(F, K, C, t, dir):
    """
    Calculate power mix of import or export on links.

    Inputs:
    F:      actual flows in the network
    K:      incidence matrix
    C:      power mix of nodes
    t:      specifying which ours of the time series to sole either an integer
            or a list of time steps.
    dir:    'export' or 'import'

    Output:
    An n by l matrix with node n's usage of links in row n and the total usage
    of link l in column l.
    """
    K = isMatrix(K)
    dim = K.shape[0]
    links = K.shape[1]

    if type(t) == int:
        timeSteps = [t]
        H = np.zeros((1, dim, links))
    elif type(t) == list:
        timeSteps = t
        H = np.zeros((len(t), dim, links))

    if dir == 'export':
        for i, t in enumerate(timeSteps):
            F = diagM(F[:, t])
            H[i] = C[i] * posM(K * F)
    elif dir == 'import':
        for i, t in enumerate(timeSteps):
            F = diagM(F[:, t])
            H[i] = C[i] * negM(K * F)
    return H
