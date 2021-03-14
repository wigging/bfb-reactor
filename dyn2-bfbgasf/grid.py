import numpy as np


def grid(params):
    """
    Calculate distance steps ∆x and one-dimensional grid points.
    """
    L = params['L']
    Lp = params['Lp']
    Ls = params['Ls']
    N1 = params['N1']
    N2 = params['N2']
    N3 = params['N3']

    # total grid points (N) and grid points to bed top (Np)
    N = N1 + N2 + N3
    Np = N1 + N2

    # dx below fuel feed (dx1), above feed (dx2), in freeboard (dx3)
    dx1 = Ls / N1
    dx2 = (Lp - Ls) / N2
    dx3 = (L - Lp) / N3

    dx = np.zeros(N)
    dx[0:N1] = dx1
    dx[N1:Np] = dx2
    dx[Np:N] = dx3

    # x points below feed (x1), above feed (x2), in freeboard (x3)
    x1 = np.arange(0, Ls + dx1 / 100, dx1)
    x2 = np.arange(Ls, Lp + dx2 / 100, dx2)
    x3 = np.arange(Lp + dx3, L + dx3 / 100, dx3)
    x = np.concatenate((x1, x2, x3))

    return dx, x
