import numpy as np


def calc_dP(dx, N, Np):
    """
    Calculate pressure drop along the reactor.
    """

    # >>> FIXME: these should be calculating
    ef = 0.48572
    rhob_g = np.full(N, 0.15)
    Mg = np.full(N, 18)
    Tg = np.full(N, 1100)
    # <<<

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

    # >>> FIXME: these should be calculated
    rhob_g = np.full(N, 0.15)
    # <<<

    # average gas mass concentration [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    return rhob_gav


def mfg_terms(params, ds, dx, mfg, N, Np, rhob_gav, sfc):
    """
    Source terms for calculating gas mass flux.
    """

    # >>> FIXME: these should be calculated
    ef = 0.48572
    mu = np.full(N, 4.1547e-05)
    rhob_g = np.full(N, 0.15)
    rhob_s = np.full(N, 1e-12)
    rhos = np.full(N, 423)
    v = np.full(N, 0.34607)
    Sg = np.full(N, 5.8362e-24)
    # <<<

    Db = params['Db']
    Lp = params['Lp']
    Ls = params['Ls']
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


def mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, N, Np, Smgg, SmgV):
    """
    Gas mass flux rate ∂ṁfg/∂t.
    """
    mfgin = params['mfgin']
    rhob_gin = params['rhob_gin']

    # gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    # initialize vector representing height along reactor
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
