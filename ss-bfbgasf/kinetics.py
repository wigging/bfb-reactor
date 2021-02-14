import numpy as np

# ----------------------------------------------------------------------------
# Biomass pyrolysis kinetic parameters
# ----------------------------------------------------------------------------

# biomass -> volatiles
Abv = 1.44e4    # frequency factor [1/s]
Ebv = 88.6e3    # activation energy [J/mol]

# biomass -> char
Abc = 7.38e5
Ebc = 106.5e3

# biomass -> tar
Abt = 4.13e6
Ebt = 112.7e3

# ----------------------------------------------------------------------------
# Gasification kinetic parameters
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Generation rates
# ----------------------------------------------------------------------------

def sb_gen(params, rhobb):
    """
    Biomass mass generation rate Sb [kg/m³⋅s].
    """
    R = params['R']
    Tgin = params['Tgin']

    Ts = Tgin
    kbv = Abv * np.exp(-Ebv / (R * Ts))
    kbc = Abc * np.exp(-Ebc / (R * Ts))
    kbt = Abt * np.exp(-Ebt / (R * Ts))
    Sb = -(kbv + kbc + kbt) * rhobb

    return Sb


def sc_gen(params, rhobb):
    """
    Char mass generation rate Sc [kg/m³⋅s].
    """
    R = params['R']
    Tgin = params['Tgin']

    Ts = Tgin
    kbc = Abc * np.exp(-Ebc / (R * Ts))
    Sc = kbc * rhobb

    return Sc


def st_gen(params, rhobb):
    """
    Tar mass generation rate St [kg/m³⋅s].
    """
    R = params['R']
    Tgin = params['Tgin']

    Ts = Tgin
    kbt = Abt * np.exp(-Ebt / (R * Ts))
    St = kbt * rhobb

    return St
