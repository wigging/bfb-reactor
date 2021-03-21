import numpy as np

# >>>
# FIXME: these variables should be calculated, assumes that N = 100
Cs = np.full(100, 1.8e-09)
Sc = np.full(100, 2.1091e-25)
Tw = np.full(100, 1100)
Xcr = np.full(100, 0.5)
ef = 0.48572
hps = np.full(100, 1.0781e-06)
qs = np.full(100, 0.00086252)
qs[-25:] = -0.00032331
qss = np.full(100, -3.6442e-19)
rhob_s = np.full(100, 1e-12)
rhos = np.full(100, 423)
umf = 0.14251
yc = np.full(100, 1e-8)
# <<<


def calc_ds(params):
    """
    Average diameter of the solid fuel particle.
    """
    db0 = params['db0']
    lb = params['lb']
    n1 = params['n1']
    rhob = params['rhob']
    rhoc = params['rhoc']
    wa = params['wa']
    wc = params['wc']

    # biomass shrinkage factor [-]
    psi = rhoc / (rhob * (wc + wa))

    # average diameter of the solid fuel particle [m]
    db = 3 * db0 * lb / (2 * lb + db0)
    ds = (1 + (1.25 * (n1 * psi * Xcr)**(1 / 3) - 1) * yc)**(-1) * db

    return ds


def calc_sfc(params, ds):
    """
    Sphericity or effective shape factor of the solid fuel particle.
    """
    lb = params['lb']

    # sphericity of solid fuel particle [-]
    sfc = 2 * ((3 / 2) * ds**2 * lb)**(2 / 3) / (ds * (ds + 2 * lb))
    return sfc


def ts_rate(params, dx, Ts, v):
    """
    Solid temperature rate ∂T𝗌/∂t.
    """
    N = params['N']
    Np = params['Np']
    N1 = params['N1']
    Tsin = params['Tsin']

    # ∂T𝗌/∂t along height of the reactor
    dtsdt = np.zeros(N)

    # below fuel inlet
    dtsdt[0:N1 - 1] = (
        -1 / (dx[0:N1 - 1] * Cs[0:N1 - 1]) * v[0:N1 - 1] * (-Ts[1:N1] + Ts[0:N1 - 1])
        + qs[0:N1 - 1] / Cs[0:N1 - 1]
        - qss[0:N1 - 1] / Cs[0:N1 - 1]
    )

    # at fuel inlet
    dtsdt[N1 - 1] = (
        -v[N1 - 1] / (dx[N1 - 1] * Cs[N1 - 1]) * (-Tsin + Ts[N1 - 1])
        + qs[N1 - 1] / Cs[N1 - 1]
        - qss[N1 - 1] / Cs[N1 - 1]
    )

    # above fuel inlet in the bed
    dtsdt[N1:Np] = (
        -1 / (dx[N1:Np] * Cs[N1:Np]) * v[N1:Np] * (Ts[N1:Np] - Ts[N1 - 1:Np - 1])
        + qs[N1:Np] / Cs[N1:Np]
        - qss[N1:Np] / Cs[N1:Np]
    )

    return dtsdt


def rhobb_rate(params, dx, mfg, rhobb, rhob_gav, Sb, v):
    """
    Biomass mass concentration rate ∂ρ̅𝖻/∂t.
    """
    Db = params['Db']
    N = params['N']
    Np = params['Np']
    N1 = params['N1']
    ms_dot = params['msdot'] / 3600

    # bed cross-sectional area [m²]
    Ab = (np.pi / 4) * (Db**2)

    # gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    # ∂ρ̅𝖻/∂t along height of the reactor
    drhobb_dt = np.zeros(N)

    # Below fuel inlet in bed
    drhobb_dt[0:N1 - 1] = -1 / dx[0:N1 - 1] * (-rhobb[1:N1] * v[1:N1] + rhobb[0:N1 - 1] * v[0:N1 - 1]) + Sb[0:N1 - 1]

    # At fuel inlet in bed
    vin = max(v[N1 - 1], ug[N1 - 1])
    rhobbin = ms_dot / (vin * Ab)
    drhobb_dt[N1 - 1] = -1 / dx[N1 - 1] * (-rhobbin * v[N1 - 1] + rhobb[N1 - 1] * v[N1 - 1] - rhobb[N1 - 2] * v[N1 - 2]) + Sb[N1 - 1]

    # Above fuel inlet in bed
    drhobb_dt[N1:Np] = -1 / dx[N1:Np] * (rhobb[N1:Np] * v[N1:Np] - rhobb[N1 - 1:Np - 1] * v[N1 - 1:Np - 1]) + Sb[N1:Np]

    return drhobb_dt


def v_rate(params, ds, dx, mfg, mu, rhobg, rhob_gav, Sb, v, x):
    """
    Solid fuel velocity rate ∂v/∂t.
    """
    Db = params['Db']
    Lp = params['Lp']
    Ls = params['Ls']
    N = params['N']
    Np = params['Np']
    cf = params['cf']
    dp = params['dp']
    e = params['e']
    ef0 = params['ef0']
    lb = params['lb']
    rhop = params['rhop']
    ugin = params['ugin']

    fw = 0.25
    g = 9.81

    # volume fraction of gas in bed and freeboard [-]
    afg = np.ones(N)
    afg[0:Np] = ef

    # density of gas along reactor axis [kg/m³]
    rhog = rhobg / afg

    # gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    Ugb = np.mean(np.append(afg[0:Np] * ug[0:Np], ugin))
    Ugb = max(ugin, Ugb)
    Dbu = 0.00853 * (1 + 27.2 * (Ugb - umf))**(1 / 3) * (1 + 6.84 * x)**1.21
    Vb = 1.285 * (Dbu / Db)**1.52 * Db

    ub = 12.51 * (Ugb - umf)**0.362 * (Dbu / Db)**0.52 * Db

    sfc = 2 * (3 / 2 * ds**2 * lb)**(2 / 3) / (ds * (ds + 2 * lb))

    Re_dc = abs(rhog) * abs(-ug - v) * ds / mu

    Cd = (
        24 / Re_dc * (1 + 8.1716 * Re_dc**(0.0964 + 0.5565 * sfc) * np.exp(-4.0655 * sfc))
        + Re_dc * 73.69 / (Re_dc + 5.378 * np.exp(6.2122 * sfc)) * np.exp(-5.0748 * sfc)
    )

    epb = (1 - ef0) * Ls / Lp
    Yb = 1 / (1 + epb * rhos / rhob_s)
    afs = Yb * (1 - ef)

    rhopb = np.zeros(N)
    rhopb[0:Np] = epb * rhop

    g0 = 1 / afg + 3 * ds * dp / (afg**2 * (dp + ds)) * (afs / ds + epb / dp)
    cs = 3 * np.pi * (1 + e) * (0.5 + cf * np.pi / 8) * (dp + ds)**2 / (rhop * dp**3 + rhos * ds**3) * afs * rhopb * g0

    Sp = Sb + Sc

    Spav = np.zeros(N)
    Spav[0:N - 1] = 0.5 * (Sp[0:N - 1] + Sp[1:N])
    Spav[N - 1] = Sp[N - 1]

    # Solid fuel velocity rate ∂v/∂t
    # ------------------------------------------------------------------------
    dvdt = np.zeros(N)

    # in the bed
    dvdt[0:Np] = (
        -1 / dx[0:Np] * v[0:Np] * (v[0:Np] - v[1:Np + 1])
        + g * (rhos[0:Np] - rhog[0:Np]) / rhos[0:Np]
        + fw * rhop * Vb[0:Np] / (dx[0:Np] * rhos[0:Np]) * (ub[0:Np] - ub[1:Np + 1])
        + (3 / 4) * (rhog[0:Np] / rhos[0:Np]) * (Cd[0:Np] / ds[0:Np]) * np.abs(-ug[0:Np] - v[0:Np]) * (-ug[0:Np] - v[0:Np])
        - cs[0:Np] * v[0:Np] * np.abs(v[0:Np])
        + Spav[0:Np] * v[0:Np] / rhos[0:Np]
    )

    return dvdt


def tp_rate(params, ds, kg, mfg, mu, Pr, rhob_g, rhob_gav, Tg, Tp, Ts,):
    """
    Bed particle temperature rate ∂Tp/∂t.
    """
    Db = params['Db']
    Lp = params['Lp']
    Ls = params['Ls']
    N = params['N']
    Np = params['Np']
    cpp = params['cpp']
    dp = params['dp']
    ef0 = params['ef0']
    ep = params['ep']
    es = params['es']
    ew = params['ew']
    phi = params['phi']
    rhop = params['rhop']
    sc = 5.67e-8

    afg = np.ones(N)
    afg[0:Np] = ef
    rhog = rhob_g / afg

    ug = mfg / rhob_gav

    epb = (1 - ef0) * Ls / Lp

    Rep = abs(rhog) * abs(ug) * dp / mu
    Nup = (7 - 10 * afg + 5 * afg**2) * (1 + 0.7 * Rep**0.2 * Pr**0.33) + (1.33 - 2.4 * afg + 1.2 * afg**2) * Rep**0.7 * Pr**0.33
    hp = 6 * epb * kg * Nup / (phi * dp**2)

    qp = (
        hp * (Tg - Tp) - 6 * es * sc * rhob_s / (rhos * ds) * (Tp**4 - Ts**4)
        + 4 / Db * epb * 1 / ((1 - ep) / (ep * epb) + (1 - ew) / ew + 1) * sc * (Tw**4 - Tp**4)
        - hps * (Tp - Ts)
    )

    rhopb = np.zeros(N)
    rhopb[0:Np] = epb * rhop

    Cp = rhopb * cpp

    # Bed particle temperature rate ∂Tp/∂t
    dtpdt = np.zeros(N)
    dtpdt[0:Np] = qp[0:Np] / Cp[0:Np]

    return dtpdt
