import numpy as np


def calc_props(params, state):
    """
    Calculate various solid phase properties.
    """
    N = params['N']
    db0 = params['db0']
    lb = params['lb']
    n1 = params['n1']
    psi = params['psi']
    rhob = params['rhob']
    rhoc = params['rhoc']

    rhob_b = state['rhob_b']
    rhob_c = state['rhob_c']
    rhob_ca = state['rhob_ca']
    Ts = state['Ts']

    # Bulk solid mass concentration ρ̅𝗌 [kg/m³]
    rhob_s = rhob_b + rhob_c

    # Char conversion factor [-]
    Xcr = np.zeros(N)

    for i in range(N):
        if rhob_ca[i] <= 0:
            Xc = 1
        else:
            Xc = abs(rhob_c[i]) / rhob_ca[i]
            Xc = min(Xc, 1)
        Xcr[i] = Xc

    # Mass fraction of char [-] and density of solid fuel particle ρ𝗌 [kg/m³]
    yc = rhob_c / rhob_s
    rhos = (yc / rhoc + (1 - yc) / rhob)**(-1)

    # Average diameter of the solid fuel particle [m]
    db = 3 * db0 * lb / (2 * lb + db0)
    ds = (1 + (1.25 * (n1 * psi * Xcr)**(1 / 3) - 1) * yc)**(-1) * db

    # Sphericity, effective shape factor, of solid fuel particle [-]
    sfc = 2 * ((3 / 2) * ds**2 * lb)**(2 / 3) / (ds * (ds + 2 * lb))

    # Biomass and char heat capacity [J/(kg⋅K)]
    cpb = (1.5 + 1e-3 * Ts) * 1e3
    cpc = (0.44 + 2e-3 * Ts - 6.7e-7 * Ts ** 2) * 1e3

    # Solid fuel mixture heat capacity [J/(kg⋅K)]
    cps = yc * cpc + (1 - yc) * cpb

    # Bed material heat capacity per volume [J/(m³⋅K)]
    Cs = np.zeros(N)

    for i in range(N):
        if rhob_b[i] == 0:
            Cs[i] = 1
        else:
            Cs[i] = rhob_s[i] * cps[i]

    return Cs, Xcr, cps, ds, rhob_s, rhos, sfc


def calc_hps(params, state, cps, ds, ef, Lp, mu, rhob_s, rhog, rhos, ug):
    """
    Calculate particle-particle heat transfer coefficient.
    """
    Db = params['Db']
    Gp = params['Gp']
    Gs = params['Gs']
    Ls = params['Ls']
    dp = params['dp']
    cpp = params['cpp']
    e = params['e']
    ef0 = params['ef0']
    gamp = params['gamp']
    gams = params['gams']
    kp = params['kp']
    ks = params['ks']
    rhop = params['rhop']

    v = state['v']
    g = 9.81

    epb = (1 - ef0) * Ls / Lp
    Yb = 1 / (1 + epb * rhos / rhob_s)
    afs = Yb * (1 - ef)

    npp = 6 * epb / (np.pi * dp**3)
    ns = 6 * afs / (np.pi * ds**3)
    vtp = g / 18 * dp**2 * (rhop - rhog) / mu
    vts = g / 18 * ds**2 * (rhos - rhog) / mu
    gTp = (2 / 15) * (1 - e)**(-1) * (ug - vtp)**2 * (dp / Db)**2
    gTs = 2 / 15 * (1 - e)**(-1) * (ug - vts)**2 * (ds / Db)**2
    Nps = (1 / 4) * npp * ns * (dp + ds)**2 * (8 * np.pi * (gTp + gTs))**0.5

    mp = (1 / 6) * rhop * np.pi * dp**3
    ms = (1 / 6) * np.pi * rhos * dp**3
    m = mp * ms / (mp + ms)

    E = (4 / 3) * ((1 - gamp**2) / Gp + (1 - gams**2) / Gs)**(-1)
    Rr = (1 / 2) * dp * ds / (dp + ds)

    hps = (
        5.36 * Nps * (m / E)**(3 / 5) * (Rr * v)**0.7
        * ((kp * rhop * cpp)**(-0.5) + (ks * rhos * cps)**(-0.5))**(-1)
    )

    return hps


def calc_qs(params, state, ds, hps, kg, mu, Pr, rhob_s, rhog, rhos, ug):
    """
    Calculate heat transfer due to gas flow and inert bed material. This
    method must be called after the `_calc_hps()` method.
    """
    es = params['es']

    v = state['v']
    Tg = state['Tg']
    Tp = state['Tp']
    Ts = state['Ts']
    sc = 5.67e-8

    Re_dc = abs(rhog) * abs(-ug - v) * ds / mu
    Nud = 2 + 0.6 * Re_dc**0.5 * Pr**0.33
    hs = Nud * kg / ds

    qs = (
        6 * hs * rhob_s / (rhos * ds) * (Tg - Ts)
        + 6 * es * sc * rhob_s / (rhos * ds) * (Tp**4 - Ts**4)
        + hps * (Tp - Ts)
    )

    return qs


def ts_rate(params, state, Cs, dx, qs, qss):
    """
    Solid temperature rate ∂T𝗌/∂t.
    """
    N = params['N']
    Np = params['Np']
    N1 = params['N1']
    Tsin = params['Tsin']

    v = state['v']
    Ts = state['Ts']

    # Solid temperature rate ∂T𝗌/∂t
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


def rhobb_rate(params, state, dx, Sb, ug):
    """
    Biomass mass concentration rate ∂ρ̅𝖻/∂t.
    """
    Ab = params['Ab']
    N = params['N']
    Np = params['Np']
    N1 = params['N1']
    ms_dot = params['msdot'] / 3600

    rhob_b = state['rhob_b']
    v = state['v']

    # ∂ρ̅𝖻/∂t along height of the reactor
    drhobb_dt = np.zeros(N)

    # Below fuel inlet in bed
    drhobb_dt[0:N1 - 1] = -1 / dx[0:N1 - 1] * (-rhob_b[1:N1] * v[1:N1] + rhob_b[0:N1 - 1] * v[0:N1 - 1]) + Sb[0:N1 - 1]

    # At fuel inlet in bed
    vin = max(v[N1 - 1], ug[N1 - 1])
    rhobbin = ms_dot / (vin * Ab)
    drhobb_dt[N1 - 1] = -1 / dx[N1 - 1] * (-rhobbin * v[N1 - 1] + rhob_b[N1 - 1] * v[N1 - 1] - rhob_b[N1 - 2] * v[N1 - 2]) + Sb[N1 - 1]

    # Above fuel inlet in bed
    drhobb_dt[N1:Np] = -1 / dx[N1:Np] * (rhob_b[N1:Np] * v[N1:Np] - rhob_b[N1 - 1:Np - 1] * v[N1 - 1:Np - 1]) + Sb[N1:Np]

    return drhobb_dt


def v_rate(params, state, afg, ds, dx, ef, Lp, mu, rhob_s, rhog, rhos, Sb, Sc, ug, umf, x):
    """
    Solid fuel velocity rate ∂v/∂t.
    """
    Db = params['Db']
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

    v = state['v']
    fw = 0.25
    g = 9.81

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


def tp_rate(params, state, afg, ds, hps, kg, Lp, mu, Pr, rhob_s, rhog, rhos, ug):
    """
    Bed particle temperature rate ∂Tp/∂t.
    """
    Db = params['Db']
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

    Tg = state['Tg']
    Tp = state['Tp']
    Ts = state['Ts']
    Tw = state['Tw']
    sc = 5.67e-8

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


def rhobc_rate(params, state, dx, Sc):
    """
    here
    """
    N = params['N']
    Np = params['Np']

    rhob_c = state['rhob_c']
    v = state['v']

    drhobc_dt = np.zeros(N)
    drhobc_dt[0] = -1 / dx[0] * (-rhob_c[1] * v[1] + rhob_c[0] * v[0]) + Sc[0]
    drhobc_dt[1:Np] = -1 / dx[1:Np] * (-rhob_c[2:Np + 1] * v[2:Np + 1] + rhob_c[1:Np] * v[1:Np]) + Sc[1:Np]

    return drhobc_dt


def rhobca_rate(params, state, dx, Sca):
    """
    Calculate char accumulation rate.
    """
    N = params['N']
    Np = params['Np']

    rhob_ca = state['rhob_ca']
    v = state['v']

    drhobca_dt = np.zeros(N)
    drhobca_dt[0] = -1 / dx[0] * (-rhob_ca[1] * v[1] + rhob_ca[0] * v[0]) + Sca[0]
    drhobca_dt[1:Np] = -1 / dx[1:Np] * (-rhob_ca[2:Np + 1] * v[2:Np + 1] + rhob_ca[1:Np] * v[1:Np]) + Sca[1:Np]

    return drhobca_dt


def tw_rate(params, state, afg, kg, Lp, mu, Pr, rhog, ug):
    """
    Calculate wall temperature rate.
    """
    Db = params['Db']
    Dwi = params['Dwi']
    Dwo = params['Dwo']
    Ls = params['Ls']
    N = params['N']
    Np = params['Np']
    N1 = params['N1']
    Tam = params['Tam']
    Uha = params['Uha']
    cpw = params['cpw']
    dp = params['dp']
    ef0 = params['ef0']
    ep = params['ep']
    ew = params['ew']
    kw = params['kw']
    phi = params['phi']
    rhow = params['rhow']

    Tg = state['Tg']
    Tp = state['Tp']
    Tw = state['Tw']
    sc = 5.67e-8

    # Calculations
    Rep = abs(rhog) * abs(ug) * dp / mu
    Nup = (7 - 10 * afg + 5 * afg**2) * (1 + 0.7 * Rep**0.2 * Pr**0.33) + (1.33 - 2.4 * afg + 1.2 * afg**2) * Rep**0.7 * Pr**0.33

    epb = (1 - ef0) * Ls / Lp
    hp = 6 * epb * kg * Nup / (phi * dp**2)
    Uhb = 1 / (4 / (np.pi * Dwi * hp) + np.pi * Dwi / (2 * kw) * np.log(Dwo / Dwi))

    qwr = np.pi * Dwi * epb / ((1 - ep) / (ep * epb) + (1 - ew) / ew + 1) * sc * (Tw**4 - Tp**4)
    qwa = (np.pi * Dwo) * Uha * (Tw - Tam)
    qwgb = (np.pi * Dwi) * Uhb * (Tw - Tg)

    Qe = 0.0 * 9.0e3
    Qwbb = Qe / Lp - qwr - qwa - qwgb
    Qwbu = - qwr - qwa - qwgb

    Qwb = np.zeros(N)
    Qwb[0:N1] = Qwbb[0:N1]
    Qwb[N1:N] = Qwbu[N1:N]

    ReD = abs(rhog) * abs(ug) * Db / mu
    Nuf = 0.023 * ReD**0.8 * Pr**0.4
    hf = Nuf * kg / Db
    Uhf = 1 / (1 / hf + np.pi * Dwi / (2 * kw) * np.log(Dwo / Dwi))
    qwgf = (np.pi * Dwi) * Uhf * (Tw - Tg)
    Qwf = -qwa - qwgf

    mw = rhow * (Dwo**2 - Dwi**2) * np.pi / 4

    # Wall temperature rate ∂T𝗐/∂t
    # ------------------------------------------------------------------------
    dtw_dt = np.zeros(N)
    dtw_dt[0:Np] = Qwb[0:Np] / (mw * cpw)
    dtw_dt[Np:N] = Qwf[Np:N] / (mw * cpw)

    return dtw_dt
