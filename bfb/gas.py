import numpy as np


def inlet_flow(params):
    """
    Inlet flows related to the gas phase.
    """
    Ls = params.Ls
    Mgin = params.Mgin
    Pa = params.Pa
    Tgin = params.Tgin
    ef0 = params.ef0
    mfgin = params.mfgin
    rhop = params.rhop
    R = 8.314  # universal gas constant (m³⋅Pa)/(mol⋅K)
    g = 9.81

    Pin = (1 - ef0) * rhop * g * Ls + Pa
    rhog_in = Pin * Mgin / (R * Tgin) * 1e-3
    rhob_gin = rhog_in
    ugin = mfgin / rhob_gin

    return ugin


def mix_props(params, rhobz, Tg):
    """
    Gas mixture properties along the reactor.
    """
    M_CH4 = params.M_CH4
    M_CO = params.M_CO
    M_CO2 = params.M_CO2
    M_H2 = params.M_H2
    M_H2O = params.M_H2O
    N = params.N

    # Mass concentrations from solver
    _, _, _, rhob_ch4, rhob_co, rhob_co2, rhob_g, rhob_h2, rhob_h2o, rhob_t = rhobz

    # here
    Amu = np.array([27.758, 3.844, 23.811, 11.811, -36.826])
    Bmu = np.array([2.120, 4.0112, 5.3944, 4.9838, 4.290]) * 1e-1
    Cmu = np.array([-0.3280, -1.4303, -1.5411, -1.0851, -0.1620]) * 1e-4

    # here
    Acp = np.array([25.399, 34.942, 29.556, 27.437, 33.933])
    Bcp = np.array([20.178, -39.957, -6.5807, 42.315, -8.4186]) * 1e-3
    Ccp = np.array([-3.8549, 19.184, 2.0130, -1.9555, 2.9906]) * 1e-5
    Dcp = np.array([3.188, -15.303, -1.2227, 0.39968, -1.7825]) * 1e-8
    Ecp = np.array([-8.7585, 39.321, 2.2617, -0.29872, 3.6934]) * 1e-12

    # here
    Ak = np.array([3.951, -0.935, 0.158, -1.200, 0.053]) * 1e-2
    Bk = np.array([4.5918, 1.4028, 0.82511, 1.0208, 0.47093]) * 1e-4
    Ck = np.array([-6.4933, 3.3180, 1.9081, -2.2403, 4.9551]) * 1e-8

    M_g = np.array([M_H2, M_CH4, M_CO, M_CO2, M_H2O])

    n = len(M_g)
    mug = np.zeros((N, n))
    cpgg = np.zeros((N, n))
    kgg = np.zeros((N, n))

    for j in range(n):
        mug[:, j] = (Amu[j] + Bmu[j] * Tg + Cmu[j] * Tg**2) * 1e-7
        cpgg[:, j] = Acp[j] + Bcp[j] * Tg + Ccp[j] * Tg**2 + Dcp[j] * Tg**3 + Ecp[j] * Tg**4
        kgg[:, j] = Ak[j] + Bk[j] * Tg + Ck[j] * Tg**2

    yH2 = rhob_h2 / rhob_g
    yH2O = rhob_h2o / rhob_g
    yCH4 = rhob_ch4 / rhob_g
    yCO = rhob_co / rhob_g
    yCO2 = rhob_co2 / rhob_g
    yg = np.column_stack((yH2, yCH4, yCO, yCO2, yH2O))

    Mg = np.zeros(N)
    mu = np.zeros(N)
    xg = np.zeros((N, n))
    cpgm = np.zeros(N)
    kg = np.zeros(N)

    for i in range(N):
        xgm = (yg[i] / M_g) / np.sum(yg[i] / M_g)
        Mgm = np.sum(xgm * M_g)
        mugm = np.sum((xgm * mug[i] * M_g**0.5)) / np.sum((xgm * M_g**0.5))
        cpgc = np.sum(xgm * cpgg[i])
        kgc = (np.sum(xgm / kgg[i]))**(-1)
        Mg[i] = Mgm
        mu[i] = mugm
        xg[i, :] = xgm
        cpgm[i] = cpgc
        kg[i] = kgc

    # ensure no zeros in array to prevent division by 0
    xg[xg == 0] = 1e-12

    yt = rhob_t / rhob_g
    cpt = -100 + 4.40 * Tg - 1.57e-3 * Tg**2
    cpgg = cpgm / Mg * 1e3
    cpg = yt * cpt + (1 - yt) * cpgg
    Pr = cpg * mu / kg

    return cpg, cpgm, kg, Mg, mu, Pr, xg


def volume_frac(params, ef):
    """
    here
    """
    N = params.N
    Ni = params.Ni

    # Volume fraction of gas in the bed and freeboard [-]
    afg = np.ones(N)
    afg[0:Ni] = ef

    return afg


def density_press(afg, rhob_g, Mg, Tg):
    """
    here
    """
    R = 8.314

    # Density of gas along the reactor [kg/m³]
    rhog = rhob_g / afg

    # Pressure of gas along reactor axis [Pa]
    P = R * rhog * Tg / Mg * 1e3

    return rhog, P


def tg_rate(params, cpg, ds, dx, ef, kg, Lp, mfg, mu, Pr, rhobz, Tz, v):
    """
    Gas temperature rate ∂T𝗀/∂t.
    """
    Db = params.Db
    Dwi = params.Dwi
    Dwo = params.Dwo
    DH_R3 = params.DH_R3
    DH_R4 = params.DH_R4
    Ls = params.Ls
    M_CH4 = params.M_CH4
    M_CO = params.M_CO
    M_CO2 = params.M_CO2
    M_H2 = params.M_H2
    M_H2O = params.M_H2O
    N = params.N
    Ni = params.Ni
    Tgin = params.Tgin
    dp = params.dp
    kw = params.kw
    ef0 = params.ef0
    phi = params.phi
    rhob = params.rhob
    rhoc = params.rhoc
    R = 8.314

    Tg, Tp, Ts, Tw = Tz

    # Mass concentrations from solver and solid phase [kg/m³]
    rhob_b, rhob_c, _, rhob_ch4, rhob_co, rhob_co2, rhob_g, rhob_h2, rhob_h2o, _ = rhobz

    # Average mass concentration of the gas [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    # Gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    # Volume fraction of gas in the bed and freeboard [-]
    afg = np.ones(N)
    afg[0:Ni] = ef

    # Density of gas along the reactor [kg/m³]
    rhog = rhob_g / afg

    # - - -
    KR3 = 312 * np.exp(-15098 / Tg) * (rhob_ch4 / M_CH4) * 1e3

    kr4 = 0.022 * np.exp(34730 / (R * Tg))

    KR4 = (
        0.278e6 * np.exp(-12560 / (R * Tg))
        * ((rhob_co / M_CO) * (rhob_h2o / M_H2O) - (rhob_co2 / M_CO2) * (rhob_h2 / M_H2) / kr4)
    )

    qgs = (DH_R4 * KR4 + DH_R3 * KR3) * 1e3

    # - - -

    Re_dc = rhog * np.abs(-ug - v) * ds / mu
    Nud = 2 + 0.6 * Re_dc**0.5 * Pr**0.33
    hs = Nud * kg / ds

    Rep = rhog * np.abs(ug) * dp / mu
    Nup = (
        (7 - 10 * afg + 5 * afg**2) * (1 + 0.7 * Rep**0.2 * Pr**0.33)
        + (1.33 - 2.4 * afg + 1.2 * afg**2) * Rep**0.7 * Pr**0.33
    )
    epb = (1 - ef0) * Ls / Lp
    hp = 6 * epb * kg * Nup / (phi * dp**2)
    Uhb = 1 / (4 / (np.pi * Dwi * hp) + np.pi * Dwi / (2 * kw) * np.log(Dwo / Dwi))

    rhob_s = rhob_b + rhob_c
    yc = rhob_c / rhob_s
    rhos = (yc / rhoc + (1 - yc) / rhob)**(-1)

    qg = -6 * hs * rhob_s / (rhos * ds) * (Tg - Ts) - hp * (Tg - Tp) + 4 / Db * Uhb * (Tw - Tg)

    # - - -

    Cg = rhob_g * cpg

    # - - -

    ReD = rhog * np.abs(ug) * Db / mu
    Nuf = 0.023 * ReD**0.8 * Pr**0.4
    hf = Nuf * kg / Db
    Uhf = 1 / (1 / hf + np.pi * Dwi / (2 * kw) * np.log(Dwo / Dwi))

    # - - -

    dtgdt = np.zeros(N)

    dtgdt[0] = -ug[0] / (dx[0]) * (Tg[0] - Tgin) + (-qgs[0] + qg[0]) / Cg[0]

    dtgdt[1:Ni] = -ug[1:Ni] / dx[1:Ni] * (Tg[1:Ni] - Tg[0:Ni - 1]) + (-qgs[1:Ni] + qg[1:Ni]) / Cg[1:Ni]

    dtgdt[Ni:N] = (
        -ug[Ni:N] / dx[Ni:N] * (Tg[Ni:N] - Tg[Ni - 1:N - 1])
        - (qgs[Ni:N] - 4 / Db * Uhf[Ni:N] * (Tw[Ni:N] - Tg[Ni:N])) / Cg[Ni:N]
    )

    return dtgdt


def mfg_rate(params, ds, dx, ef, Lp, mfg, Mg, mu, rhobz, Sg, Tg, v):
    """
    Gas mass flux rate ∂ṁfg/∂t.
    """
    Ab = params.Ab
    Db = params.Db
    Ls = params.Ls
    Mgin = params.Mgin
    N = params.N
    N1 = params.N1
    Ni = params.Ni
    Pa = params.Pa
    Tgin = params.Tgin
    dp = params.dp
    ef0 = params.ef0
    lb = params.lb
    mfgin = params.mfgin
    ms_dot = params.ms_dot / 3600
    phi = params.phi
    rhob = params.rhob
    rhoc = params.rhoc
    rhop = params.rhop

    g = 9.81
    R = 8.314

    # Mass concentrations from solver [kg/m³]
    rhob_b, rhob_c, _, _, _, _, rhob_g, _, _, _ = rhobz

    # Average mass concentration of the gas [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    # Gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    # Calculations
    # ------------------------------------------------------------------------
    Pin = (1 - ef0) * rhop * g * Ls + Pa
    rhog_in = Pin * Mgin / (R * Tgin) * 1e-3
    rhob_gin = rhog_in

    # Volume fraction of gas in bed and freeboard [-]
    afg = np.ones(N)
    afg[0:Ni] = ef

    # Density of gas along reactor axis [kg/m³]
    rhog = rhob_g / afg

    # Pressure of gas along reactor axis [Pa]
    P = R * rhog * Tg / Mg * 1e3

    DP = -afg[0:N - 1] / dx[0:N - 1] * (P[1:N] - P[0:N - 1])

    epb = (1 - ef0) * Ls / Lp

    sfc = 2 * ((3 / 2) * ds**2 * lb)**(2 / 3) / (ds * (ds + 2 * lb))

    vin = max(v[N1 - 1], ug[N1 - 1])
    rhobbin = ms_dot / (vin * Ab)

    rhob_s = rhob_b + rhob_c

    rhosbav = np.zeros(N)
    rhosbav[0:N - 1] = 0.5 * (rhob_s[0:N - 1] + rhob_s[1:N])
    rhosbav[N - 1] = 0.5 * (rhobbin + rhob_s[N - 1])

    Re_dc = rhog * np.abs(-ug - v) * ds / mu

    Cd = (
        24 / Re_dc * (1 + 8.1716 * Re_dc**(0.0964 + 0.5565 * sfc) * np.exp(-4.0655 * sfc))
        + Re_dc * 73.69 / (Re_dc + 5.378 * np.exp(6.2122 * sfc)) * np.exp(-5.0748 * sfc)
    )

    Reg = rhob_gav * ug * Db / mu

    fg = np.zeros(N)
    for i in range(N):
        if Reg[i] <= 2300:
            fg[i] = 16 / Reg[i]
        else:
            fg[i] = 0.079 / Reg[i]**0.25

    yc = rhob_c / rhob_s
    rhos = (yc / rhoc + (1 - yc) / rhob)**(-1)

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

    # Gas mass flux rate ∂mfg/∂t
    # ------------------------------------------------------------------------
    dmfgdt = np.zeros(N)

    # at gas inlet, bottom of reactor
    Cmf1 = -1 / (2 * dx[0]) * ((mfg[1] + mfg[0]) * ug[0] - (mfg[0] + mfgin) * mfgin / rhob_gin)
    Smg1 = Cmf1 + SmgV[0]
    dmfgdt[0] = Smg1 + DP[0]

    # in the bed
    Smg = Cmf + SmgV[1:N - 1]
    dmfgdt[1:Ni + 1] = Smg[0:Ni] + DP[1:Ni + 1]

    # in the bed top and in the freeboard
    Smgf = Cmf[Ni - 1:] + Smgg[Ni:N - 1] * ug[Ni:N - 1] + DP[Ni:N - 1]
    dmfgdt[Ni:N - 1] = Smgf

    # at top of reactor
    SmgfN = -1 / dx[N - 1] * mfg[N - 1] * (ug[N - 1] - ug[N - 2]) + Smgg[N - 1] * ug[N - 1]
    dmfgdt[N - 1] = SmgfN

    return dmfgdt


def rhobg_rate(params, dx, mfg, Sg):
    """
    here
    """
    N = params.N
    mfgin = params.mfgin

    drhobg_dt = np.zeros(N)
    drhobg_dt[0] = -(mfg[0] - mfgin) / dx[0] + Sg[0]
    drhobg_dt[1:N] = -(mfg[1:N] - mfg[0:N - 1]) / dx[1:N] + Sg[1:N]

    return drhobg_dt


def rhobh2o_rate(params, dx, mfg, rhob_g, rhob_h2o, Sh2o, ugin):
    """
    here
    """
    Mgin = params.Mgin
    N = params.N
    Pin = params.Pin
    Tgin = params.Tgin
    R = 8.314

    yH2O = rhob_h2o / rhob_g

    rhog_in = Pin * Mgin / (R * Tgin) * 1e-3
    rhob_h2oin = rhog_in

    drhobh2o_dt = np.zeros(N)
    drhobh2o_dt[0] = -(yH2O[0] * mfg[0] - rhob_h2oin * ugin) / dx[0] + Sh2o[0]
    drhobh2o_dt[1:N] = -(yH2O[1:N] * mfg[1:N] - yH2O[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sh2o[1:N]

    return drhobh2o_dt


def rhobh2_rate(params, dx, mfg, rhob_g, rhob_h2, Sh2, ugin):
    """
    here
    """
    N = params.N

    yH2 = rhob_h2 / rhob_g
    rhob_h2in = 0

    drhobh2_dt = np.zeros(N)
    drhobh2_dt[0] = -(yH2[0] * mfg[0] - rhob_h2in * ugin) / dx[0] + Sh2[0]
    drhobh2_dt[1:N] = -(yH2[1:N] * mfg[1:N] - yH2[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sh2[1:N]

    return drhobh2_dt


def rhobch4_rate(params, dx, mfg, rhob_ch4, rhob_g, Sch4, ugin):
    """
    here
    """
    N = params.N

    yCH4 = rhob_ch4 / rhob_g
    rhob_ch4in = 0

    drhobch4_dt = np.zeros(N)
    drhobch4_dt[0] = -(yCH4[0] * mfg[0] - rhob_ch4in * ugin) / dx[0] + Sch4[0]
    drhobch4_dt[1:N] = -(yCH4[1:N] * mfg[1:N] - yCH4[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sch4[1:N]

    return drhobch4_dt


def rhobco_rate(params, dx, mfg, rhob_co, rhob_g, Sco, ugin):
    """
    here
    """
    N = params.N

    yCO = rhob_co / rhob_g
    rhob_coin = 0

    drhobco_dt = np.zeros(N)
    drhobco_dt[0] = -(yCO[0] * mfg[0] - rhob_coin * ugin) / dx[0] + Sco[0]
    drhobco_dt[1:N] = -(yCO[1:N] * mfg[1:N] - yCO[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sco[1:N]

    return drhobco_dt


def rhobco2_rate(params, dx, mfg, rhob_co2, rhob_g, Sco2, ugin):
    """
    here
    """
    N = params.N

    yCO2 = rhob_co2 / rhob_g
    rhob_co2in = 0

    drhobco2_dt = np.zeros(N)
    drhobco2_dt[0] = -(yCO2[0] * mfg[0] - rhob_co2in * ugin) / dx[0] + Sco2[0]
    drhobco2_dt[1:N] = -(yCO2[1:N] * mfg[1:N] - yCO2[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sco2[1:N]

    return drhobco2_dt


def rhobt_rate(params, dx, mfg, rhob_t, rhob_g, St, ugin):
    """
    here
    """
    N = params.N

    yt = rhob_t / rhob_g
    rhob_tin = 0

    drhobt_dt = np.zeros(N)
    drhobt_dt[0] = -(yt[0] * mfg[0] - rhob_tin * ugin) / dx[0] + St[0]
    drhobt_dt[1:N] = -(yt[1:N] * mfg[1:N] - yt[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + St[1:N]

    return drhobt_dt
