import numpy as np

# Pyrolysis kinetic parameters
# A is frequency factor [1/s]
# E is activation energy [J/mol]
# bv is biomass -> volatiles
# bc is biomass -> char
# bt is biomass -> tar
Abv = 1.44e4
Ebv = 88.6e3
Abc = 7.38e5
Ebc = 106.5e3
Abt = 4.13e6
Ebt = 112.7e3

# Molecular weight [g/mol]
M_C = 12
M_CH4 = 16
M_CO = 28
M_CO2 = 44
M_H2 = 2
M_H2O = 18

# Tar cracking mass fractions
rCH4 = 0.08841
rCO = 0.56333
rCO2 = 0.11093
rH2 = 0.01733

# Heats of reaction
DH_R2 = -74.8
DH_R3 = 206
DH_R4 = -41.2
DH_R5 = 172
DH_R6 = 131
Dhpy = 64e3


def calc_sgen(params, P, Ts, Tg, Xcr, rhob_b, rhob_c, rhob_ch4, rhob_co, rhob_co2, rhob_h2, rhob_h2o, rhob_t, xg):
    """
    Mass generation rates for gas phase and solid phase.
    """
    N = params['N']
    Ni = params['Np']
    wH2O = params['wH2O']
    R = 8.314

    # Biomass mass generation rate Sb [kg/(m³⋅s)]
    kbc = Abc * np.exp(-Ebc / (R * Ts))
    kbt = Abt * np.exp(-Ebt / (R * Ts))
    kbv = Abv * np.exp(-Ebv / (R * Ts))
    Sb = -(kbv + kbc + kbt) * rhob_b

    # Carbon mass generation rate Sc [kg/(m³⋅s)]
    Tss = 0.5 * (Ts + Tg)
    KR2 = 6.11 * 1e3 * np.exp(-80333 / (R * Tss)) * (rhob_h2 / M_H2) * (rhob_c / M_C)

    k5r1 = 3.6e5 * np.exp(-20130 / Tss)
    k5r2 = 4.15e3 * np.exp(-11420 / Tss)
    KR5 = k5r1 / (1 + xg[2] * (k5r2 * xg[3])**(-1)) * (rhob_c / M_C) * 1e3

    for i in range(N):
        if xg[3, i] == 0:
            KR5[i] = 0

    k6r1 = 1.25e5 * np.exp(-28000 / Tss)
    k6r2 = 3.26e-4
    k6r3 = 0.313 * np.exp(-10120 / Tss)
    KR6 = k6r1 * xg[4] / (1 / P + k6r3 * xg[4] + k6r2 * xg[0]) * Xcr * (rhob_c / M_C) * 1e3
    Sc = kbc * rhob_b - (KR2 + KR5 + KR6) * M_C * 1e-3

    # Accumulated carbon mass generation rate Sca [kg/(m³⋅s)]
    Sca = kbc * rhob_b

    # Mass fractions of volatile gases
    m0v = [1.34e-16, 1.80e7, 2.48e3, 4.43e5]
    Tsv = np.mean(Ts[0:Ni])
    b0v = [5.727, -1.871, -0.696, -1.495]
    yv = m0v * Tsv**b0v
    Mv = [M_H2, M_CO, M_CO2, M_CH4]
    xv = yv * Mv / np.sum(yv * Mv)
    vH2 = xv[0]
    vCO = xv[1]
    vCO2 = xv[2]
    vCH4 = xv[3]

    # H₂ mass generation rate Sh2 [kg/(m³⋅s)]
    kbv = Abv * np.exp(-Ebv / (R * Ts))
    Sbv = kbv * rhob_b
    Sbv[Ni:N] = 0

    Tss = 0.5 * (Ts + Tg)
    KR2 = 6.11 * 1e3 * np.exp(-80333 / (R * Tss)) * (rhob_h2 / M_H2) * (rhob_c / M_C)
    KR3 = 312 * np.exp(-15098 / Tg) * (rhob_ch4 / M_CH4) * 1e3

    kr4 = 0.022 * np.exp(34730 / (R * Tg))
    KR4 = 0.278e6 * np.exp(-12560 / (R * Tg)) * ((rhob_co / M_CO) * (rhob_h2o / M_H2O) - (rhob_co2 / M_CO2) * (rhob_h2 / M_H2) / kr4)

    k6r1 = 1.25e5 * np.exp(-28000 / Tss)
    k6r2 = 3.26e-4
    k6r3 = 0.313 * np.exp(-10120 / Tss)
    KR6 = k6r1 * xg[4] / (1 / P + k6r3 * xg[4] + k6r2 * xg[0]) * Xcr * (rhob_c / M_C) * 1e3

    kt = 9.55e4 * np.exp(-93.37e3 / (R * Tg))
    Sh2 = vH2 * Sbv + rH2 * kt * rhob_t + (-2 * KR2 + 3 * KR3 + KR4 + KR6) * M_H2 * 1e-3

    # CH₄ mass generation rate Sch4 [kg/(m³⋅s)]
    Sch4 = vCH4 * Sbv + rCH4 * kt * rhob_t + (KR2 - KR3) * M_CH4 * 1e-3

    # CO mass generation rate Sco [kg/(m³⋅s)]
    k5r1 = 3.6e5 * np.exp(-20130 / Tss)
    k5r2 = 4.15e3 * np.exp(-11420 / Tss)
    KR5 = k5r1 / (1 + xg[2] * (k5r2 * xg[3])**(-1)) * (rhob_c / M_C) * 1e3

    for i in range(N):
        if xg[3, i] == 0:
            KR5[i] = 0

    Sco = vCO * Sbv + rCO * kt * rhob_t + (-KR4 + KR3 + 2 * KR5 + KR6) * M_CO * 1e-3

    # CO₂ mass generation rate Sco2 [kg/(m³⋅s)]
    Sco2 = vCO2 * Sbv + rCO2 * kt * rhob_t + (KR4 - KR5) * M_CO2 * 1e-3

    # H₂O mass generation rate Sh2o [kg/(m³⋅s)]
    Sh2o = -wH2O * Sb - (KR3 + KR4 + KR6) * M_H2O * 1e-3

    # Tar mass generation rate St [kg/(m³⋅s)]
    kbt = Abt * np.exp(-Ebt / (R * Ts))
    Sbt = kbt * rhob_b
    Sbt[Ni:N] = 0
    St = Sbt - kt * rhob_t

    # Overall gas mass generation rate Sg [kg/(m³⋅s)]
    Sg = Sh2 + Sch4 + Sco + Sco2 + Sh2o + St

    return Sb, Sc, Sca, Sh2, Sh2o, Sch4, Sco, Sco2, Sg, St


def calc_qgs(rhob_ch4, rhob_co, rhob_co2, rhob_h2, rhob_h2o, Tg):
    """
    here
    """
    R = 8.314

    KR3 = 312 * np.exp(-15098 / Tg) * (rhob_ch4 / M_CH4) * 1e3

    kr4 = 0.022 * np.exp(34730 / (R * Tg))

    KR4 = (
        0.278e6 * np.exp(-12560 / (R * Tg))
        * ((rhob_co / M_CO) * (rhob_h2o / M_H2O) - (rhob_co2 / M_CO2) * (rhob_h2 / M_H2) / kr4)
    )

    qgs = (DH_R4 * KR4 + DH_R3 * KR3) * 1e3

    return qgs


def calc_qss(params, P, rhob_c, rhob_h2, Sb, Tg, Ts, Xcr, xg):
    """
    Heat generation term qss for solid temperature rate equation ∂T𝗌/∂t.
    Represents net heat generation from solid phase gasification and
    pyrolysis reactions. This method must be called after the `_calc_sb()`
    method.
    """
    N = params['N']
    R = 8.314

    Tss = 0.5 * (Ts + Tg)
    KR2 = 6.11 * 1e3 * np.exp(-80333 / (R * Tss)) * (rhob_h2 / M_H2) * (rhob_c / M_C)

    k5r1 = 3.6e5 * np.exp(-20130 / Tss)
    k5r2 = 4.15e3 * np.exp(-11420 / Tss)
    KR5 = k5r1 / (1 + xg[2] * (k5r2 * xg[3])**(-1)) * (rhob_c / M_C) * 1e3

    for i in range(N):
        if xg[3, i] == 0:
            KR5[i] = 0

    k6r1 = 1.25e5 * np.exp(-28000 / Tss)
    k6r2 = 3.26e-4
    k6r3 = 0.313 * np.exp(-10120 / Tss)
    KR6 = k6r1 * xg[4] / (1 / P + k6r3 * xg[4] + k6r2 * xg[0]) * Xcr * (rhob_c / M_C) * 1e3

    qss = (DH_R2 * KR2 + DH_R5 * KR5 + DH_R6 * KR6) * 1e3 + Dhpy * Sb

    return qss
