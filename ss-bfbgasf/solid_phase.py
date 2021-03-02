import numpy as np


def mfs_rhobb_inlet(params, ugin):
    """
    h
    """
    D = params['D']
    msin = params['msin']

    A = (np.pi / 4) * D**2
    mfsin = msin / A

    vin = ugin
    rhobbin = mfsin / vin

    return mfsin, rhobbin


def alpha_fracs(params, rhos, rhosb):
    """
    Void fractions α𝗉 [-] and α𝗌 [-].
    """
    ef0 = params['ef0']
    ef = ef0

    afp = 1 - ef

    Yb = 1 / (1 + (1 - ef) * rhos / rhosb)
    afs = Yb * (1 - ef)

    return afp, afs


def cp_solid(Ts, yc):
    """
    Heat capacity of the solid fuel.
    """
    cpb = (1.5 + (1e-3) * Ts) * 1e3
    cpc = (0.44 + (2e-3) * Ts - (6.7e-7) * Ts**2) * 1e3
    cps = yc * cpc + (1 - yc) * cpb

    return cps


def betaps_momentum(params, afs, ds, mfsin, rhos, rhosb, v):
    """
    Solid fuel to inert bed material momentum transfer coefficient β𝗉𝗌
    [N⋅s/m⁴]. Momentum transfer due to collision with inert bed particles.
    """
    cf = params['cf']
    dp = params['dp']
    e = params['e']
    ef0 = params['ef0']
    rhop = params['rhop']

    ef = ef0
    epb = 1 - ef0
    rhopb = epb * rhop

    g0 = 1 / ef + 3 * ds * dp / (ef**2 * (dp + ds)) * (afs / ds + epb / dp)
    cs = 3 * np.pi * (1 + e) * (0.5 + cf * np.pi / 8) * (dp + ds)**2 / (rhop * dp**3 + rhos * ds**3) * rhosb * rhopb * g0
    Smps = cs * abs(v)

    return Smps


def ds_rhos_fuel(params, ya, yb, yc):
    """
    Average diameter of the solid fuel particle d𝗌 [m] and solid fuel density ρ𝗌 [kg/m³].
    """
    db0 = params['db0']
    lb = params['lb']
    rhoa = params['rhoa']
    rhoc = params['rhoc']
    rhobio = params['rhobio']
    wa = params['wa']
    wc = params['wc']

    dbio = 3 * db0 * lb / (2 * lb + db0)
    da = (wa * rhobio / rhoa)**(1 / 3) * dbio
    dc = (wc * rhobio / rhoc)**(1 / 3) * dbio

    rhob = (1 - wa) / (1 / rhobio - wa / rhoa)
    db = ((1 - wa) * (rhobio / rhob))**(1 / 3) * dbio

    ds = (ya / da + yc / dc + yb / db)**(-1)
    rhos = (ya / rhoa + yc / rhoc + yb / rhob)**(-1)

    return ds, rhos


def hps_coeff(params, afp, afs, cps, ds, rhogin, rhos, ug, yc, v):
    """
    Particle-particle heat transfer coefficient h𝗉𝗌 [W/(m³⋅K)].
    """
    D = params['D']
    Gp = params['Gp']
    Gs = params['Gs']
    e = params['e']
    cpp = params['cpp']
    dp = params['dp']
    g = params['g']
    kp = params['kp']
    ks = params['ks']
    mugin = params['mugin']
    rhop = params['rhop']
    vp = params['gamp']
    vs = params['gams']

    m = (np.pi / 6) * ((rhos * rhop * ds**3 * dp**3) / (rhos * ds**3 + rhop * dp**3))

    s = (1 - vs**2) / Gs
    t = (1 - vp**2) / Gp
    E = (4 / 3) / (s + t)

    mu = mugin
    rhog = rhogin
    vtp = g / 18 * dp**2 * (rhop - rhog) / mu
    vts = g / 18 * ds**2 * (rhos - rhog) / mu

    oms = (2 / 15) * ((ug - vts)**2 / (1 - e)) * (ds / D)**2
    omp = (2 / 15) * ((ug - vtp)**2 / (1 - e)) * (dp / D)**2

    a = afp * afs * (ds + dp)**2
    b = (ds**3) * (dp**3) * ((rhos * cps * ks)**(-1 / 2) + (rhop * cpp * kp)**(-1 / 2))
    d = (ds * dp) / (2 * (ds + dp))
    hps = 4.88 * (a / b) * (m / E)**(3 / 5) * (d * v)**(7 / 10) * np.sqrt(8 * np.pi * (oms + omp))

    return hps


def hwp_conv(params, afp, rhogin):
    """
    Convective heat transfer coefficient h𝗐𝗉 [W/(m²⋅K)] between bulk inert
    particles and the reactor walls.
    """
    cpp = params['cpp']
    dp = params['dp']
    ef0 = params['ef0']
    g = params['g']
    kp = params['kp']
    rhop = params['rhop']

    ef = ef0
    rhog = rhogin

    cpg = 1100
    kg = 0.03

    S = 0.0282 * afp**(-0.59)
    ec = 1 - 1.23 * (1 - ef)**0.54

    x = (1 - ec) * (1 - (kg / kp))
    y = kp / kg + 0.28 * ec**(0.63 * (kg / kp)**0.18)
    kc = kg * (1 + (x / y))

    Cc = (1 - ec) * rhop * cpp + ec * rhog * cpg
    Lc = 0.0178 * rhop**0.596
    uc = 0.75 * np.sqrt((rhop / rhog) * g * dp)

    tc = Lc / uc

    a = ((np.pi * tc) / (4 * kc * Cc))**0.5
    b = (S * dp) / kg
    hwp = 1 / (a + b)

    return hwp


def kr_coeff(params, afp):
    """
    Effective radiation coefficient K𝗋 [1/m].
    """
    D = params['D']
    ep = params['ep']
    ew = params['ew']

    Kr = (4 / D) * ((1 - ep) / (ep * afp**2) + 1 / ew)**(-1)

    return Kr


def sfc_fuel(params):
    """
    Mean sphericity φ𝐬 [-] of the solid fuel particles.
    """
    db0 = params['db0']
    lb = params['lb']

    dbio = 3 * db0 * lb / (2 * lb + db0)
    sfc = 2 * (3 / 2 * dbio**2 * lb)**(2 / 3) / (dbio * (dbio + 2 * lb))

    return sfc


def tp_inert(params, afp, afs, ds, hgp, hps, hwp, Kr, Tp, Ts):
    """
    Solid inert bed temperature Tp [K].
    """
    D = params['D']
    N = params['N']
    Tgin = params['Tgin']
    dp = params['dp']
    es = params['es']
    phi = params['phi']
    sc = params['sc']

    Tg = np.full(N, Tgin)
    Tw = np.full(N, Tgin)

    Xp = (6 / (phi * dp)) * afp * hgp + hps + (4 / D) * afp * hwp

    Sp = -(6 / ds) * afs * es * sc * (Tp**4 - Ts**4) + Kr * sc * (Tw**4 - Tp**4)
    Yp = (6 / (phi * dp)) * afp * hgp * Tg + hps * Ts + (4 / D) * afp * hwp * Tw + Sp

    Tp = Yp / Xp

    return Tp


def y_fracs(rhoab, rhobb, rhocb):
    """
    Mass fractions for biomass, char, and ash.
    """
    ya = rhoab / (rhobb + rhocb + rhoab)
    yb = rhobb / (rhobb + rhocb + rhoab)
    yc = rhocb / (rhobb + rhocb + rhoab)

    return ya, yb, yc


def rhoab_coeffs(dz, Sa, v):
    """
    Coefficients a, b, c, for the ash mass concentration matrix.
    """
    a = v
    b = v
    c = dz * Sa

    return a, b, c


def rhobb_coeffs(params, dz, rhobbin, Sb, v):
    """
    Coefficients a, b, c for the biomass mass concentration matrix.
    """
    N = params['N']

    a = v
    b = v
    c = dz * Sb
    c[N - 1] = c[N - 1] + b[N - 1] * rhobbin

    return a, b, c


def rhobb_coeffs2(params, dz, kb, mfsin, v):
    """
    h
    """
    N = params['N']

    a = v + (dz * kb)
    b = v
    c = np.zeros(N)
    c[N - 1] = c[N - 1] + mfsin

    return a, b, c


def rhocb_coeffs(dz, Sc, v):
    """
    Coefficients a, b, c for the char mass concentration matrix.
    """
    a = v
    b = v
    c = dz * Sc

    return a, b, c


def ts_coeffs(params, afs, cps, ds, dz, hgs, hps, rhosb, Sb, Tp, Ts, v):
    """
    Coefficients a, b, c for solid fuel temperature matrix.
    """
    N = params['N']
    Tgin = params['Tgin']
    Tsin = params['Tsin']
    es = params['es']
    sc = params['sc']

    Hpyr = 64000

    Tg = np.full(N, Tgin)

    a = v + (dz / (rhosb * cps)) * ((6 / ds) * afs * hgs + hps)
    b = v
    c = (dz / (rhosb * cps)) * ((6 / ds) * afs * hgs * Tg + (6 / ds) * es * sc * (Tp**4 - Ts**4) + hps * Tp - 0 - Sb * Hpyr)

    vin = v[N - 1]
    c[N - 1] = c[N - 1] + vin * Tsin

    return a, b, c


def v_coeffs(params, dz, rhos, Ms_res, Smgs, Smps, Sss, ug, v):
    """
    Coefficients a, b, c for solid fuel velocity matrix.
    """
    N = params['N']

    a = v + dz / rhos * (Smgs + Smps - Sss)
    b = v
    c = dz / rhos * (Ms_res - Smgs * ug)

    vin = v[N - 1]
    c[N - 1] = c[N - 1] + b[N - 1] * vin

    return a, b, c


def v_coeffs2(params, dz, Fb, rhogin, rhos, Smgs, Smps, Sss, ug, v):
    """
    Coefficients a, b, c for solid fuel velocity matrix.
    """
    N = params['N']
    g = params['g']

    rhog = rhogin
    Ms_res = g * (rhos - rhog) + Fb

    a = rhos * v + 2 * dz * (Smgs + Smps - Sss)
    b = rhos * v
    c = 2 * dz * (Ms_res - Smgs * ug)

    vin = v[N - 1]
    c[N - 1] = c[N - 1] + b[N - 1] * vin

    return a, b, c
