import numpy as np


def rhog_inlet(params):
    """
    Gas density at inlet.
    """
    L = params['L']
    Mgin = params['Mgin']
    Pa = params['Pa']
    R = params['R']
    Tgin = params['Tgin']
    ef0 = params['ef0']
    g = params['g']
    rhop = params['rhop']

    Pin = (1 - ef0) * rhop * g * L + Pa
    rhogin = Pin * Mgin / (R * Tgin) * 1e-3

    return rhogin


def mfg_ug_inlet(params, rhogin):
    """
    Gas mass flux and gas velocity as inlet.
    """
    D = params['D']
    mgin = params['mgin']

    A = (np.pi / 4) * D**2  # cross-section area of the bed [m²]
    mfgin = mgin / A        # gas mass flux
    ugin = mfgin / rhogin   # gas velocity [m/s]

    return mfgin, ugin


def umf_bed(params, rhogin):
    """
    Minimum fluidization velocity U𝗆𝖿 [m/s] for the inert bed material.
    """
    dp = params['dp']
    g = params['g']
    mugin = params['mugin']
    rhop = params['rhop']

    Ar = dp**3 * rhogin * (rhop - rhogin) * g / mugin**2
    Rem = -33.67 + (33.67**2 + 0.0408 * Ar)**0.5
    umf = Rem * mugin / (rhogin * dp)

    return umf


def fb_prime(params, ug, ugin, umf, z):
    """
    Force Fʙ́ [N/m³] exerted on the fuel particles by the inert bed material
    due to bubble flow.
    """
    D = params['D']
    N = params['N']
    ef0 = params['ef0']
    emf = params['emf']
    rhop = params['rhop']

    ef = ef0
    thetaw = 0.25

    ugmean = np.mean(np.concatenate(([ugin], ef * ug)))
    usf = max(ugin, ugmean)

    db = 0.00853 * (1 + 27.2 * (usf - umf))**(1 / 3) * (1 + 6.84 * z)**1.21
    Vb = 1.285 * (db / D)**1.52 * D

    ub = 12.51 * (usf - umf)**0.362 * (db / D)**0.52 * D
    ubu = np.concatenate((ub[2:N + 1], [ug[N - 1]]))
    ubl = ub[1:N + 1]

    dz = z[1] - z[0] / 2
    Fb = -(1 - emf) * rhop * thetaw * Vb[1:] * (ubu - ubl) / dz

    return Fb


def betagp_momentum(params, afp, rhogin, ug):
    """
    Gas to inert bed material momentum transfer coefficient β𝗀𝗉 [N⋅s/m⁴].
    """
    ef0 = params['ef0']
    mugin = params['mugin']
    dp = params['dp']
    phi = params['phi']

    rhog = rhogin
    ef = ef0
    mu = mugin

    a = 150 * ((afp * (1 - ef)) / (ef * (phi * dp)**2)) * mu
    b = 1.75 * (afp / (phi * dp)) * rhog * abs(ug)
    Smgp = a + b

    return Smgp


def betags_momentum(params, ds, sfc, rhogin, ug, v):
    """
    Gas to solid fuel momentum transfer coefficient β𝗀𝗌 [N⋅s/m⁴]. Momentum
    transfer due to drag by the gas.
    """
    mugin = params['mugin']

    rhog = rhogin
    mu = mugin
    Re_dc = rhog * abs(-ug - v) * ds / mu

    Cd = (
        24 / Re_dc
        * (1 + 8.1716 * Re_dc**(0.0964 + 0.5565 * sfc) * np.exp(-4.0655 * sfc))
        + Re_dc * 73.69 / (Re_dc + 5.378 * np.exp(6.2122 * sfc))
        * np.exp(-5.0748 * sfc)
    )

    Smgs = 3 / 4 * rhog * (Cd / ds) * abs(ug + v)

    return Smgs


def fg_factor(params, rhogin, ug):
    """
    Wall friction factor f𝗀 [-].
    """
    D = params['D']
    N = params['N']
    mugin = params['mugin']

    rhogb = rhogin
    Reg = rhogb * ug * D / mugin

    fg = np.zeros(N)

    for i in range(N):
        if Reg[i] <= 2300:
            fg[i] = 16 / Reg[i]
        else:
            fg[i] = 0.079 / Reg[i]**0.25

    return fg


def mfg_coeffs(params, afg, dz, fg, mfgin, rhogin, Sgs, Smgp, Smgs, ug, ugin, v):
    """
    Coefficients a, b, c, d for gas mass flux matrix.
    """
    D = params['D']
    N = params['N']
    R = params['R']
    Mgin = params['Mgin']
    Tgin = params['Tgin']
    ef0 = params['ef0']
    g = params['g']
    rhop = params['rhop']

    rhog = rhogin
    rhogb = rhogin

    Tg = np.full(N, Tgin)
    Mg = np.full(N, Mgin)
    P = R * rhog * Tg / Mg * 1e3

    dp = np.zeros(len(P))
    dp[:-1] = afg / dz * np.diff(P)

    epb = 1 - ef0
    Mg_res = g * (afg * epb * rhop - rhogb) - dp

    a = np.concatenate(([ugin], ug[0:N - 1]))

    b1 = ug[0] - ugin - 2 * dz / rhogb * (afg * Smgs[0] - Smgp[0] + 2 * fg[0] / D * abs(ug[0]) + Sgs)
    b_inner = ug[1:N] - ug[0:N - 1] - 2 * dz / rhogb * (afg * Smgs[1:N] - Smgp[1:N] + 2 * fg[1:N] / D * abs(ug[1:N]) + Sgs)
    b = np.concatenate(([b1], b_inner))

    c = ug
    b[N - 1] = b[N - 1] + c[N - 1]

    d = 2 * dz * (afg * Smgs * v + Mg_res)
    d[0] = d[0] + a[0] * mfgin

    return a, b, c, d


def mfg_coeffs2(params, afs, dz, fg, mfgin, rhogin, Sgs, Smgp, Smgs, ug, ugin, v):
    """
    Coefficients a, b, c, d for gas mass flux matrix. (Version 2)
    """
    D = params['D']
    N = params['N']
    R = params['R']
    Mgin = params['Mgin']
    Tgin = params['Tgin']
    ef0 = params['ef0']
    g = params['g']
    rhop = params['rhop']

    ef = ef0
    rhog = rhogin
    rhogb = rhogin

    Tg = np.full(N, Tgin)
    Mg = np.full(N, Mgin)
    P = R * rhog * Tg / Mg * 1e3

    dp = np.zeros(len(P))
    dp[:-1] = np.diff(P)

    a = np.concatenate(([ugin], ug[0:N - 1]))

    b1 = ug[0] - ugin - (2 * dz / rhogb) * (afs[0] * Smgs[0] - Smgp[0] + Sgs)
    binner = ug[1:N] - ug[0:N - 1] - (2 * dz / rhogb) * (afs[1:N] * Smgs[1:N] - Smgp[1:N] + Sgs)
    b = np.concatenate(([b1], binner))

    c = ug
    b[N - 1] = b[N - 1] + c[N - 1]

    d1 = g * (ef * (1 - ef) * rhop - rhogb)
    d2 = 2 * fg * rhogb / D * ug * abs(ug)
    d = 2 * dz * (d1 + afs * Smgs * v - d2 - ef * dp / dz)
    d[0] = d[0] + ugin * mfgin

    return a, b, c, d
