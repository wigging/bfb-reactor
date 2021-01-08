import numpy as np


def grid(params, Lp):
    """
    Calculate distance steps ∆x and one-dimensional grid.
    """
    L = params.L
    Lf0 = params.Lf0
    N = params.N
    N1 = params.N1
    N2 = params.N2
    N3 = params.N3
    Ni = params.Ni

    # Distance steps
    dx1 = Lf0 / N1
    dx2 = (Lp - Lf0) / N2
    dx3 = (L - Lp) / N3

    dx = np.zeros(N)
    dx[0:N1] = dx1
    dx[N1:Ni] = dx2
    dx[Ni:N] = dx3

    # One-dimensional grid
    # Add small value to end-value so the end-value is included in range
    x1 = np.arange(0, Lf0 + dx1, dx1)
    x2 = np.arange(Lf0, Lp + dx2/100, dx2)
    x3 = np.arange(Lp + dx3, L + dx3/100, dx3)
    x = np.concatenate((x1, x2, x3))

    return dx, x
