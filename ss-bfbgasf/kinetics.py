import numpy as np


def sb_gen(params, rhobb):
    """
    Biomass mass generation rate Sb [kg/m³⋅s].
    """
    R = params['R']
    Tgin = params['Tgin']

    Ts = Tgin

    A0bv = 1.44e4
    Ebv = 88.6e3

    A0bc = 7.38e5
    Ebc = 106.5e3

    A0bt = 4.13e6
    Ebt = 112.7e3

    kbv = A0bv * np.exp(-Ebv / (R * Ts))
    kbc = A0bc * np.exp(-Ebc / (R * Ts))
    kbt = A0bt * np.exp(-Ebt / (R * Ts))

    Sb = -(kbv + kbc + kbt) * rhobb

    return Sb


def sc_gen(params, rhobb):
    """
    Char mass generation rate Sc [kg/m³⋅s].
    """
    R = params['R']
    Tgin = params['Tgin']

    Ts = Tgin

    A0bc = 7.38e5
    Ebc = 106.5e3

    kbc = A0bc * np.exp(-Ebc / (R * Ts))

    Sc = kbc * rhobb

    return Sc
