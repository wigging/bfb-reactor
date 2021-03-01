import numpy as np

# Molecular weight [g/mol]
M_C = 12
M_CH4 = 16
M_CO = 28
M_CO2 = 44
M_H2 = 2
M_H2O = 18

# Pyrolysis kinetic parameters
# A is frequency factor [1/s]
# E is activation energy [J/mol]
Abv = 1.44e4    # biomass -> volatiles
Ebv = 88.6e3

Abc = 7.38e5    # biomass -> char
Ebc = 106.5e3

Abt = 4.13e6    # biomass -> tar
Ebt = 112.7e3


def kb_rate(params, Ts):
    """
    Total biomass pyrolysis rate constant k𝖻 [1/s].
    """
    R = params['R']

    kbv = Abv * np.exp(-Ebv / (R * Ts))
    kbc = Abc * np.exp(-Ebc / (R * Ts))
    kbt = Abt * np.exp(-Ebt / (R * Ts))
    kb = (kbv + kbc + kbt)

    return kb


def sa_gen(params, rhocb, Ts):
    """
    Ash mass generation rate Sa [kg/m³⋅s].
    """
    Pa = params['Pa']
    R = params['R']
    wa = params['wa']
    wc = params['wc']

    p = Pa
    Xc = 0.5
    rhobh2 = 0.01
    xCO = 0.01
    xCO2 = 0.01
    xH2O = 0.01
    xH2 = 0.01

    k51 = 1.25e5 * np.exp(-28000 / Ts)
    k52 = 3.26e-4
    k53 = 0.313 * np.exp(-10120 / Ts)
    r5 = (k51 * xH2O) / (1 / p + k52 * xH2 + k53 * xH2O) * (1 - Xc) * (rhocb / M_C * 1000)

    k61 = 3.6e5 * np.exp(-20130 / Ts)
    k62 = 4.15e3 * np.exp(-11420 / Ts)
    r6 = k61 / (1 + xCO / (k62 * xCO2)) * (rhocb / M_C * 1000)

    r7 = 6.11e-3 * np.exp(-80333 / (R * Ts)) * (rhobh2 / M_H2 * 1000) * (rhocb / M_C * 1000)

    Sa = (wa / wc) * (r5 + r6 + r7) * M_C * (1 / 1000)

    return Sa


def sb_gen(params, rhobb, Ts):
    """
    Biomass mass generation rate Sb [kg/m³⋅s].
    """
    R = params['R']

    kbv = Abv * np.exp(-Ebv / (R * Ts))
    kbc = Abc * np.exp(-Ebc / (R * Ts))
    kbt = Abt * np.exp(-Ebt / (R * Ts))
    Sb = -(kbv + kbc + kbt) * rhobb

    return Sb


def sc_gen(params, rhobb, rhocb, Ts):
    """
    Char mass generation rate Sc [kg/m³⋅s].
    """
    Pa = params['Pa']
    R = params['R']

    p = Pa
    Xc = 0.5
    rhobh2 = 0.01
    xCO = 0.01
    xCO2 = 0.01
    xH2O = 0.01
    xH2 = 0.01

    kbc = Abc * np.exp(-Ebc / (R * Ts))

    k51 = 1.25e5 * np.exp(-28000 / Ts)
    k52 = 3.26e-4
    k53 = 0.313 * np.exp(-10120 / Ts)
    r5 = (k51 * xH2O) / (1 / p + k52 * xH2 + k53 * xH2O) * (1 - Xc) * (rhocb / M_C * 1000)

    k61 = 3.6e5 * np.exp(-20130 / Ts)
    k62 = 4.15e3 * np.exp(-11420 / Ts)
    r6 = k61 / (1 + xCO / (k62 * xCO2)) * (rhocb / M_C * 1000)

    r7 = 6.11e-3 * np.exp(-80333 / (R * Ts)) * (rhobh2 / M_H2 * 1000) * (rhocb / M_C * 1000)

    Sc = kbc * rhobb - (r5 + r6 + r7) * M_C * (1 / 1000)

    return Sc


def st_gen(params, rhobb, Ts):
    """
    Tar mass generation rate St [kg/m³⋅s].
    """
    R = params['R']

    kbt = Abt * np.exp(-Ebt / (R * Ts))
    St = kbt * rhobb

    return St
