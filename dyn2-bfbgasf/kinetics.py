import numpy as np


# >>>
# FIXME: these variables should be calculated, assumes that N = 100
Xcr = np.full(100, 0.5)
ef = 0.48572
rhob_c = np.full(100, 1e-8)
rhob_ch4 = np.full(100, 1e-8)
rhob_co = np.full(100, 1e-8)
rhob_co2 = np.full(100, 1e-8)
rhob_h2 = np.full(100, 1e-8)
# <<<


# Pyrolysis kinetic parameters
# A is frequency factor [1/s]
# E is activation energy [J/mol]
Abv = 1.44e4    # biomass -> volatiles
Ebv = 88.6e3
Abc = 7.38e5    # biomass -> char
Ebc = 106.5e3
Abt = 4.13e6    # biomass -> tar
Ebt = 112.7e3

# Molecular weight [g/mol]
M_C = 12
M_CH4 = 16
M_CO = 28
M_CO2 = 44
M_H2 = 2
M_H2O = 18


def calc_sb(rhobb, Ts):
    """
    Calculate the biomass mass generation rate Sb [kg/(m³⋅s)].
    """
    R = 8.314

    kbv = Abv * np.exp(-Ebv / (R * Ts))
    kbc = Abc * np.exp(-Ebc / (R * Ts))
    kbt = Abt * np.exp(-Ebt / (R * Ts))
    Sb = -(kbv + kbc + kbt) * rhobb

    return Sb


def calc_h2o(params, Mg, rhob_g, rhob_h2o, Sb, Tg, Ts, xg):
    """
    Calculate the H₂O mass generation rate Sh2o [kg/(m³⋅s)].
    """
    N = params['N']
    Np = params['Np']
    wH2O = params['wH2O']
    R = 8.314

    Tss = 0.5 * (Ts + Tg)

    afg = np.ones(N)
    afg[0:Np] = ef
    rhog = rhob_g / afg
    P = R * rhog * Tg / Mg * 1e3

    KR3 = 312 * np.exp(-15098 / Tg) * (rhob_ch4 / M_CH4) * 1e3

    kr4 = 0.022 * np.exp(34730 / (R * Tg))
    KR4 = 0.278e6 * np.exp(-12560 / (R * Tg)) * ((rhob_co / M_CO) * (rhob_h2o / M_H2O) - (rhob_co2 / M_CO2) * (rhob_h2 / M_H2) / kr4)

    k6r1 = 1.25e5 * np.exp(-28000 / Tss)
    k6r2 = 3.26e-4
    k6r3 = 0.313 * np.exp(-10120 / Tss)
    KR6 = k6r1 * xg[4] / (1 / P + k6r3 * xg[4] + k6r2 * xg[3]) * Xcr * (rhob_c / M_C) * 1e3

    Sh2o = -wH2O * Sb - (KR3 + KR4 + KR6) * M_H2O * 1e-3

    return Sh2o
