import numpy as np

# Pyrolysis kinetic parameters
# A is frequency factor [1/s]
# E is activation energy [J/mol]
Abv = 1.44e4    # biomass -> volatiles
Ebv = 88.6e3
Abc = 7.38e5    # biomass -> char
Ebc = 106.5e3
Abt = 4.13e6    # biomass -> tar
Ebt = 112.7e3


def calc_sb(rhobb, Ts):
    """
    Calculate the biomass mass generation rate Sb [kg/m³⋅s].
    """
    R = 8.314

    kbv = Abv * np.exp(-Ebv / (R * Ts))
    kbc = Abc * np.exp(-Ebc / (R * Ts))
    kbt = Abt * np.exp(-Ebt / (R * Ts))
    Sb = -(kbv + kbc + kbt) * rhobb

    return Sb
