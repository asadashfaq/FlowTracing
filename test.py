from __future__ import division
import sys
import numpy as np
from numpy import matrix
import matplotlib.pyplot as plt
from functions import *
from tracer import *

"""
Test consistency with the up/down stream flow tracing
"""

if len(sys.argv) < 2:
    raise Exception('Not enough inputs!')
else:
    task = str(sys.argv[1:])

names = ['AT', 'FI', 'NL', 'BA', 'FR', 'NO', 'BE', 'GB', 'PL', 'BG', 'GR',
         'PT', 'CH', 'HR', 'RO', 'CZ', 'HU', 'RS', 'DE', 'IE', 'SE', 'DK',
         'IT', 'SI', 'ES', 'LU', 'SK', 'EE', 'LV', 'LT']

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
    print normCols(C[0]) * abs(negM(diagM(P)))

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
    C = normCols(C)
    E = C[0] * abs(negM(diagM(P[:, t])))
    #print N['power_mix'][0, :, t]
    #print E[0, :]
    #print sum(N['power_mix'][0, :, t])
    #print sum(E[0, :])
    #print N['power_mix'][0, :, 100] - C[0, 0, :]

    pmim = N['power_mix'][:, :, t]
    pmex = N['power_mix_ex'][:, :, t]
    nodes = pmim.shape[0]

    shift = .3
    width = .5

    for n in range(nodes):
        #if P[n, t] > 0:
        #    pmim[n, n] = 0
        #    print ' +  , ', sum(E[:, n]) - P[n, t]
        #if P[n, t] < 0:
        #    pmex[n, n] = 0
        #    print '- , ', sum(E[n, :]) + P[n, t]

        plt.figure(figsize=(13, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(nodes), np.reshape(E[:, n], (1, nodes)).tolist()[0], width, edgecolor='none', color='SteelBlue')
        plt.bar(np.arange(shift, nodes + shift, 1), pmim[n, :], width, edgecolor='none', color='LightSteelBlue')
        plt.xticks(np.arange((width + shift) * .5, nodes + (width + shift) * .5, 1), names, rotation=75, fontsize=10)
        plt.legend(('Matrix formulation', 'Up/down stream'), loc='best')
        plt.title('Import', fontsize=12)
        plt.ylabel('MW')
        plt.ylim(ymin=0)

        plt.subplot(1, 2, 2)
        plt.bar(range(nodes), E[n, :].tolist()[0], width, edgecolor='none', color='SteelBlue')
        plt.bar(np.arange(shift, nodes + shift, 1), pmex[n, :], width, edgecolor='none', color='LightSteelBlue')
        plt.xticks(np.arange((width + shift) * .5, nodes + (width + shift) * .5, 1), names, rotation=75, fontsize=10)
        plt.legend(('Matrix formulation', 'Up/down stream'), loc='best')
        plt.title('Export', fontsize=12)
        plt.ylabel('MW')
        plt.ylim(ymin=0)

        plt.suptitle(names[n] + " (" + str(int(round(P[n, t]))) + "), t = " + str(t), fontsize=14)
        plt.savefig('./figures/nodes/' + str(n) + '.png', bbox_inches='tight')
        plt.close()
