from __future__ import division
import sys
import numpy as np
from numpy import matrix
from functions import *
from tracer import *

"""
Test consistency with the up/down stream flow tracing
"""

if len(sys.argv) < 2:
    raise Exception('Not enough inputs!')
else:
    task = str(sys.argv[1:])

if 'simple' in task:
    """
    Test the simple example from Rolando's thesis
    """
    K = matrix([[1, 0, 0], [0, 1, 0], [-1, -1, 1], [0, 0, -1]])
    P = np.array([6, 6, -4, -8])
    P = np.reshape(P, (4, 1))
    F = np.array([6, 6, 8])
    F = np.reshape(F, (3, 1))
    t = 0
    if 'export' in task:
        C = nodeMix(F, K, P, t, dir='export')
    elif 'import' in task:
        C = nodeMix(F, K, P, t, dir='import')
    print C[0]
    print normCols(C[0])

if 'linear' in task:
    """
    Compare to the up/down stream flow tracing for localised flow
    """
    F = np.load('./input/linear-flows.npy')
    K = np.load('./input/K.npy')
    N = np.load('./input/linear_pm.npz', mmap_mode='r')
    P = N['mismatch'] + N['balancing']
    t = [100]
    C = nodeMix(F, K, P, t, dir='export')
    print N['power_mix'][0, :, 100]
    print C[0, 0, :]
    print sum(N['power_mix'][0, :, 100])
    print sum(C[0, 0, :])
    #print N['power_mix'][0, :, 100] - C[0, 0, :]
