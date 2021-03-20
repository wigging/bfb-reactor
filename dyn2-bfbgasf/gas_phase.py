import numpy as np


# >>>
# FIXME: these variables should be calculated, assumes that N = 100
Sg = np.full(100, 5.8362e-24)
Tp = np.zeros(100)
Tp[:75] = 1100
Tw = np.full(100, 1100)
ef = 0.48572
qgs = np.zeros(100)
rhob_h2 = np.full(100, 1e-8)
rhob_h2o = np.full(100, 0.15)
rhob_ch4 = np.full(100, 1e-8)
rhob_co = np.full(100, 1e-8)
rhob_co2 = np.full(100, 1e-8)
rhob_t = np.full(100, 1e-8)
rhob_g = np.full(100, 0.15)
rhob_s = np.full(100, 1e-12)
rhos = np.full(100, 423)
# v = np.full(100, 0.34607)
# <<<

# Molecular weight [g/mol]
M_CH4 = 16
M_CO = 28
M_CO2 = 44
M_H2 = 2
M_H2O = 18

# Gas viscosity coefficients (see Table S1 in Agu 2019)
# coefficients listed in order of CH4, CO, CO2, H2, H2O
Amu = np.array([3.844, 23.811, 11.811, 27.758, -36.826])
Bmu = np.array([4.0112, 5.3944, 4.9838, 2.120, 4.290]) * 1e-1
Cmu = np.array([-1.4303, -1.5411, -1.0851, -0.3280, -0.1620]) * 1e-4

# Gas specific heat capacity coefficients (see Table S2 in Agu 2019)
# coefficients listed in order of CH4, CO, CO2, H2, H2O
Acp = np.array([34.942, 29.556, 27.437, 25.399, 33.933])
Bcp = np.array([-39.957, -6.5807, 42.315, 20.178, -8.4186]) * 1e-3
Ccp = np.array([19.184, 2.0130, -1.9555, -3.8549, 2.9906]) * 1e-5
Dcp = np.array([-15.303, -1.2227, 0.39968, 3.188, -1.7825]) * 1e-8
Ecp = np.array([39.321, 2.2617, -0.29872, -8.7585, 3.6934]) * 1e-12

# Gas thermal conductivity coefficients (see Table S3 in Agu 2019)
# coefficients listed in order of CH4, CO, CO2, H2, H2O
Ak = np.array([-0.935, 0.158, -1.200, 3.951, 0.053]) * 1e-2
Bk = np.array([1.4028, 0.82511, 1.0208, 4.5918, 0.47093]) * 1e-4
Ck = np.array([3.3180, 1.9081, -2.2403, -6.4933, 4.9551]) * 1e-8


def calc_dP(params, dx, Mg, Tg):
    """
    Calculate pressure drop along the reactor.
    """
    N = params['N']
    Np = params['Np']
    R = 8.314

    # volume fraction of gas in bed and freeboard [-]
    afg = np.ones(N)
    afg[0:Np] = ef

    # density of gas along reactor axis [kg/m³]
    rhog = rhob_g / afg

    # pressure along reactor axis [Pa]
    P = R * rhog * Tg / Mg * 1e3

    # pressure drop along the reactor
    DP = -afg[0:N - 1] / dx[0:N - 1] * (P[1:N] - P[0:N - 1])

    return DP


def calc_rhobgav(N):
    """
    Calculate the average gas mass concentration.
    """

    # average gas mass concentration [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    return rhob_gav


def calc_mix_props(Tg):
    """
    Calculate gas mixture properties along the reactor.
    """

    # molecular weights
    M = np.array([[M_CH4], [M_CO], [M_CO2], [M_H2], [M_H2O]])

    # mass fractions
    rhobx = np.array([rhob_ch4, rhob_co, rhob_co2, rhob_h2, rhob_h2o])
    yx = rhobx / rhob_g

    # mole fractions
    sumYM = np.sum(yx, axis=0) / (M_CH4 + M_CO + M_CO2 + M_H2 + M_H2O)
    xg = (yx / M) / sumYM

    # viscosity
    mux = (Amu[:, None] + Bmu[:, None] * Tg + Cmu[:, None] * Tg**2) * 1e-7

    # thermal conductivity
    kx = Ak[:, None] + Bk[:, None] * Tg + Ck[:, None] * Tg**2

    # heat capacity
    cpx = Acp[:, None] + Bcp[:, None] * Tg + Ccp[:, None] * Tg**2 + Dcp[:, None] * Tg**3 + Ecp[:, None] * Tg**4

    # calculate mixture properties
    Mg = sum(xg * M)
    mu = sum(xg * mux * M**0.5) / sum(xg * M**0.5)
    cpgm = sum(xg * cpx)
    kg = (sum(xg / kx))**(-1)

    cpt = -100 + 4.40 * Tg - 1.57e-3 * Tg**2
    cpgg = cpgm / Mg * 1e3
    yt = rhob_t / rhob_g
    cpg = yt * cpt + (1 - yt) * cpgg
    Pr = cpg * mu / kg

    return Mg, Pr, cpg, cpgm, kg, mu, xg


def mfg_terms(params, ds, dx, mfg, mu, rhob_gav, sfc, v):
    """
    Source terms for calculating gas mass flux.
    """
    Db = params['Db']
    Lp = params['Lp']
    Ls = params['Ls']
    N = params['N']
    Np = params['Np']
    N1 = params['N1']
    dp = params['dp']
    ef0 = params['ef0']
    msdot = params['msdot'] / 3600
    phi = params['phi']
    rhop = params['rhop']

    g = 9.81

    epb = (1 - ef0) * Ls / Lp

    # volume fraction of gas in bed and freeboard [-]
    afg = np.ones(N)
    afg[0:Np] = ef

    # density of gas along reactor axis [kg/m³]
    rhog = rhob_g / afg

    # gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    # drag coefficient
    Re_dc = rhog * np.abs(-ug - v) * ds / mu

    Cd = (
        24 / Re_dc * (1 + 8.1716 * Re_dc**(0.0964 + 0.5565 * sfc) * np.exp(-4.0655 * sfc))
        + Re_dc * 73.69 / (Re_dc + 5.378 * np.exp(6.2122 * sfc)) * np.exp(-5.0748 * sfc)
    )

    # bed cross-sectional area [m²]
    Ab = (np.pi / 4) * (Db**2)

    vin = max(v[N1 - 1], ug[N1 - 1])
    rhobbin = msdot / (vin * Ab)

    # average bulk density of fuel particles in each reactor cell [kg/m³]
    rhosbav = np.zeros(N)
    rhosbav[0:N - 1] = 0.5 * (rhob_s[0:N - 1] + rhob_s[1:N])
    rhosbav[N - 1] = 0.5 * (rhobbin + rhob_s[N - 1])

    Reg = rhob_gav * ug * Db / mu

    fg = np.zeros(N)

    for i in range(N):
        if Reg[i] <= 2300:
            fg[i] = 16 / Reg[i]
        else:
            fg[i] = 0.079 / Reg[i]**0.25

    Sgav = np.zeros(N)
    Sgav[0:N - 1] = 0.5 * (Sg[0:N - 1] + Sg[1:N])
    Sgav[N - 1] = Sg[N - 1]

    Smgg = Sgav
    Smgp = 150 * epb**2 * mu / (ef * (phi * dp)**2) + 1.75 * epb / (phi * dp) * rhog * ug
    Smgs = (3 / 4) * rhosbav * (rhog / rhos) * (Cd / ds) * np.abs(-ug - v)
    SmgG = g * (epb * afg * rhop - rhob_gav)
    SmgF = 2 / Db * fg * rhob_gav * np.abs(ug) * ug
    SmgV = SmgG + Smgs * (ug + v) - (Smgp - Smgg) * ug - SmgF

    Cmf = -1 / (2 * dx[1:N - 1]) * ((mfg[2:N] + mfg[1:N - 1]) * ug[1:N - 1] - (mfg[1:N - 1] + mfg[0:N - 2]) * ug[0:N - 2])

    return Cmf, Smgg, SmgV


def mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, Smgg, SmgV):
    """
    Gas mass flux rate ∂ṁfg/∂t.
    """
    N = params['N']
    Np = params['Np']
    mfgin = params['mfgin']
    rhob_gin = params['rhob_gin']

    # gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    # ∂ṁfg/∂t along height of the reactor
    dmfgdt = np.zeros(N)

    # at gas inlet, bottom of reactor
    Cmf1 = -1 / (2 * dx[0]) * ((mfg[1] + mfg[0]) * ug[0] - (mfg[0] + mfgin) * mfgin / rhob_gin)
    Smg1 = Cmf1 + SmgV[0]
    dmfgdt[0] = Smg1 + DP[0]

    # in the bed
    Smg = Cmf + SmgV[1:N - 1]
    dmfgdt[1:Np + 1] = Smg[0:Np] + DP[1:Np + 1]

    # in the bed top and in the freeboard
    Smgf = Cmf[Np - 1:] + Smgg[Np:N - 1] * ug[Np:N - 1] + DP[Np:N - 1]
    dmfgdt[Np:N - 1] = Smgf

    # at top of reactor
    SmgfN = -1 / dx[N - 1] * mfg[N - 1] * (ug[N - 1] - ug[N - 2]) + Smgg[N - 1] * ug[N - 1]
    dmfgdt[N - 1] = SmgfN

    return dmfgdt


def tg_rate(params, cpg, ds, dx, kg, mfg, mu, Pr, rhob_gav, Tg, Ts, v):
    """
    Gas temperature rate ∂T𝗀/∂t.
    """
    Db = params['Db']
    Dwi = params['Dwi']
    Dwo = params['Dwo']
    Lp = params['Lp']
    Ls = params['Ls']
    N = params['N']
    Np = params['Np']
    Tgin = params['Tgin']
    dp = params['dp']
    ef0 = params['ef0']
    kw = params['kw']
    phi = params['phi']

    # volume fraction of gas in bed and freeboard [-]
    afg = np.ones(N)
    afg[0:Np] = ef

    # density of gas along reactor axis [kg/m³]
    rhog = rhob_g / afg

    # gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    Re_dc = abs(rhog) * abs(-ug - v) * ds / mu
    Nud = 2 + 0.6 * Re_dc**0.5 * Pr**0.33
    hs = Nud * kg / ds

    Rep = abs(rhog) * np.abs(ug) * dp / mu
    Nup = (
        (7 - 10 * afg + 5 * afg**2) * (1 + 0.7 * Rep**0.2 * Pr**0.33)
        + (1.33 - 2.4 * afg + 1.2 * afg**2) * Rep**0.7 * Pr**0.33
    )
    epb = (1 - ef0) * Ls / Lp
    hp = 6 * epb * kg * Nup / (phi * dp**2)
    Uhb = 1 / (4 / (np.pi * Dwi * hp) + np.pi * Dwi / (2 * kw) * np.log(Dwo / Dwi))

    qg = -6 * hs * rhob_s / (rhos * ds) * (Tg - Ts) - hp * (Tg - Tp) + 4 / Db * Uhb * (Tw - Tg)

    # - - -

    Cg = rhob_g * cpg

    # - - -

    ReD = abs(rhog) * np.abs(ug) * Db / mu
    Nuf = 0.023 * ReD**0.8 * Pr**0.4
    hf = Nuf * kg / Db
    Uhf = 1 / (1 / hf + np.pi * Dwi / (2 * kw) * np.log(Dwo / Dwi))

    # - - -

    # ∂T𝗀/∂t along height of the reactor [K]
    dtgdt = np.zeros(N)

    dtgdt[0] = -ug[0] / (dx[0]) * (Tg[0] - Tgin) + (-qgs[0] + qg[0]) / Cg[0]

    dtgdt[1:Np] = -ug[1:Np] / dx[1:Np] * (Tg[1:Np] - Tg[0:Np - 1]) + (-qgs[1:Np] + qg[1:Np]) / Cg[1:Np]

    dtgdt[Np:N] = (
        -ug[Np:N] / dx[Np:N] * (Tg[Np:N] - Tg[Np - 1:N - 1])
        - (qgs[Np:N] - 4 / Db * Uhf[Np:N] * (Tw[Np:N] - Tg[Np:N])) / Cg[Np:N]
    )

    return dtgdt
