import numpy as np


# ----------------------------------------------------------------------------
# Solid phase inlet
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Solid phase calculations
# ----------------------------------------------------------------------------

def ds_rhos_fuel(params, rhoba, rhobb, rhobc):
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

    ya = rhoba / (rhobb + rhobc + rhoba)
    yc = rhobc / (rhobb + rhobc + rhoba)
    yb = rhobb / (rhobb + rhobc + rhoba)

    ds = (ya / da + yc / dc + yb / db)**(-1)

    rhos = (ya / rhoa + yc / rhoc + yb / rhob)**(-1)

    return ds, rhos


def sfc_fuel(params):
    """
    Mean sphericity φ𝐬 [-] of the solid fuel particles.
    """
    db0 = params['db0']
    lb = params['lb']

    dbio = 3 * db0 * lb / (2 * lb + db0)
    sfc = 2 * (3 / 2 * dbio**2 * lb)**(2 / 3) / (dbio * (dbio + 2 * lb))

    return sfc


def ms_res(params, Fb, rhogin, rhos):
    """
    here
    """
    g = params['g']

    rhog = rhogin
    Ms_res = g * (rhos - rhog) + Fb

    return Ms_res


def betaps_momentum(params, afg, ds, mfsin, rhos, rhobs, v):
    """
    Solid fuel to inert bed material momentum transfer coefficient β𝗉𝗌
    [N⋅s/m⁴]. Momentum transfer due to collision with inert bed particles.
    """
    cf = params['cf']
    dp = params['dp']
    e = params['e']
    ef0 = params['ef0']
    rhop = params['rhop']

    epb = 1 - ef0

    Yb = 1 / (1 + epb * rhos / rhobs)
    afs = Yb * (1 - ef0)

    rhopb = epb * rhop

    g0 = 1 / afg + 3 * ds * dp / (afg**2 * (dp + ds)) * (afs / ds + epb / dp)
    cs = 3 * np.pi * (1 + e) * (0.5 + cf * np.pi / 8) * (dp + ds)**2 / (rhop * dp**3 + rhos * ds**3) * rhobs * rhopb * g0
    Smps = cs * abs(v)

    return Smps


# ----------------------------------------------------------------------------
# Solid phase coefficients
# ----------------------------------------------------------------------------

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


def rhoba_coeffs(dz, Sa, v):
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


def rhobc_coeffs(dz, Sc, v):
    """
    Coefficients a, b, c for the char mass concentration matrix.
    """
    a = v
    b = v
    c = dz * Sc

    return a, b, c
