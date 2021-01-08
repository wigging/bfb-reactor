import numpy as np


def char_conv(params, rhob_c, rhob_ca):
    """
    Char conversion factor.
    """
    Xcr = np.zeros(params.N)

    for i in range(params.N):
        if rhob_ca[i] <= 0:
            Xc = 1
        else:
            Xc = abs(rhob_c[i]) / rhob_ca[i]
            Xc = min(Xc, 1)
        Xcr[i] = Xc

    return Xcr


def particle_diam(params, rhob_b, rhob_c, Xcr):
    """
    Solid particle diameter.
    """
    db = params.db
    n1 = params.n1
    psi = params.psi

    rhob_s = rhob_b + rhob_c
    yc = rhob_c / rhob_s
    ds = (1 + (1.25 * (n1 * psi * Xcr)**(1 / 3) - 1) * yc)**(-1) * db
    return ds


def heat_cap(params, rhob_b, rhob_c, Ts):
    """
    Heat capacity c̅𝗉 [J/(kg⋅K)] for the solids.
    """
    N = params.N

    rhob_s = rhob_b + rhob_c
    yc = rhob_c / rhob_s

    # Biomass heat capacity [J/(kg⋅K)]
    cpb = (1.5 + 1e-3 * Ts) * 1e3

    # Char heat capacity [J/(kg⋅K)]
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

    return cpb, cpc, cps, Cs


def heat_coeff(params, cps, ds, ef, Lp, mfg, mu, rhog, rhob_b, rhob_c, rhob_g, v):
    """
    here
    """
    Db = params.Db
    Gp = params.Gp
    Gs = params.Gs
    Ls = params.Ls
    N = params.N
    dp = params.dp
    cpp = params.cpp
    e = params.e
    ef0 = params.ef0
    gamp = params.gamp
    gams = params.gams
    kp = params.kp
    ks = params.ks
    rhob = params.rhob
    rhoc = params.rhoc
    rhop = params.rhop
    g = 9.81

    rhob_s = rhob_b + rhob_c
    yc = rhob_c / rhob_s
    rhos = (yc / rhoc + (1 - yc) / rhob)**(-1)

    epb = (1 - ef0) * Ls / Lp
    Yb = 1 / (1 + epb * rhos / rhob_s)
    afs = Yb * (1 - ef)

    # Average mass concentration of the gas [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    # Gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

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


def ts_rate(params, Cs, dx, ds, kg, hps, mfg, mu, P, Pr, rhog, rhobz, Sb, Tz, v, Xcr, xg):
    """
    Solid temperature rate ∂T𝗌/∂t [K/s].
    """
    DH_R2 = params.DH_R2
    DH_R5 = params.DH_R5
    DH_R6 = params.DH_R6
    Dhpy = params.Dhpy
    M_C = params.M_C
    M_H2 = params.M_H2
    N = params.N
    N1 = params.N1
    Ni = params.Ni
    Tsin = params.Tsin
    es = params.es
    rhob = params.rhob
    rhoc = params.rhoc

    R = 8.314
    sc = 5.67e-8

    rhob_b, rhob_c, _, _, _, _, rhob_g, rhob_h2, _, _ = rhobz
    Tg, Tp, Ts, _ = Tz

    # - - -

    # Average mass concentration of the gas [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    # Gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    Re_dc = rhog * np.abs(-ug - v) * ds / mu
    Nud = 2 + 0.6 * Re_dc**0.5 * Pr**0.33
    hs = Nud * kg / ds

    rhob_s = rhob_b + rhob_c
    yc = rhob_c / rhob_s
    rhos = (yc / rhoc + (1 - yc) / rhob)**(-1)

    qs = (
        6 * hs * rhob_s / (rhos * ds) * (Tg - Ts)
        + 6 * es * sc * rhob_s / (rhos * ds) * (Tp**4 - Ts**4)
        + hps * (Tp - Ts)
    )

    # - - -

    Tss = 0.5 * (Ts + Tg)
    KR2 = 6.11 * 1e3 * np.exp(-80333 / (R * Tss)) * (rhob_h2 / M_H2) * (rhob_c / M_C)

    k5r1 = 3.6e5 * np.exp(-20130 / Tss)
    k5r2 = 4.15e3 * np.exp(-11420 / Tss)
    KR5 = k5r1 / (1 + xg[:, 2] * (k5r2 * xg[:, 3])**(-1)) * (rhob_c / M_C) * 1e3
    for i in range(N):
        if xg[i, 3] == 0:
            KR5[i] = 0

    k6r1 = 1.25e5 * np.exp(-28000 / Tss)
    k6r2 = 3.26e-4
    k6r3 = 0.313 * np.exp(-10120 / Tss)
    KR6 = k6r1 * xg[:, 4] / (1 / P + k6r3 * xg[:, 4] + k6r2 * xg[:, 0]) * Xcr * (rhob_c / M_C) * 1e3

    qss = (DH_R2 * KR2 + DH_R5 * KR5 + DH_R6 * KR6) * 1e3 + Dhpy * Sb

    # - - -

    dtsdt = np.zeros(N)

    dtsdt[0:N1 - 1] = (
        -1 / (dx[0:N1 - 1] * Cs[0:N1 - 1]) * v[0:N1 - 1] * (-Ts[1:N1] + Ts[0:N1 - 1])
        + qs[0:N1 - 1] / Cs[0:N1 - 1]
        - qss[0:N1 - 1] / Cs[0:N1 - 1]
    )

    dtsdt[N1 - 1] = (
        -v[N1 - 1] / (dx[N1 - 1] * Cs[N1 - 1]) * (-Tsin + Ts[N1 - 1])
        + qs[N1 - 1] / Cs[N1 - 1]
        - qss[N1 - 1] / Cs[N1 - 1]
    )

    dtsdt[N1:Ni] = (
        -1 / (dx[N1:Ni] * Cs[N1:Ni]) * v[N1:Ni] * (Ts[N1:Ni] - Ts[N1 - 1:Ni - 1])
        + qs[N1:Ni] / Cs[N1:Ni]
        - qss[N1:Ni] / Cs[N1:Ni]
    )

    return dtsdt


def tw_rate(params, afg, kg, mfg, mu, Lp, Pr, rhog, rhob_g, Tg, Tp, Tw):
    """
    Calculate wall temperature rate.
    """
    Db = params.Db
    Dwo = params.Dwo
    Dwi = params.Dwi
    Ls = params.Ls
    N = params.N
    Ni = params.Ni
    N1 = params.N1
    Tam = params.Tam
    Uha = params.Uha
    dp = params.dp
    cpw = params.cpw
    ef0 = params.ef0
    ep = params.ep
    ew = params.ew
    kw = params.kw
    phi = params.phi
    rhow = params.rhow
    sc = 5.67e-8

    # Average mass concentration of the gas [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    # Gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    Rep = rhog * np.abs(ug) * dp / mu
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

    ReD = rhog * np.abs(ug) * Db / mu
    Nuf = 0.023 * ReD**0.8 * Pr**0.4
    hf = Nuf * kg / Db
    Uhf = 1 / (1 / hf + np.pi * Dwi / (2 * kw) * np.log(Dwo / Dwi))
    qwgf = (np.pi * Dwi) * Uhf * (Tw - Tg)
    Qwf = -qwa - qwgf

    mw = rhow * (Dwo**2 - Dwi**2) * np.pi / 4

    dtw_dt = np.zeros(N)
    dtw_dt[0:Ni] = Qwb[0:Ni] / (mw * cpw)
    dtw_dt[Ni:N] = Qwf[Ni:N] / (mw * cpw)

    return dtw_dt


def tp_rate(params, afg, ds, hps, kg, Lp, mfg, mu, Pr, rhog, rhob_b, rhob_c, rhob_g, Tg, Tp, Ts, Tw):
    """
    here
    """
    Db = params.Db
    Ls = params.Ls
    N = params.N
    Ni = params.Ni
    cpp = params.cpp
    dp = params.dp
    ef0 = params.ef0
    ep = params.ep
    es = params.es
    ew = params.ew
    phi = params.phi
    rhob = params.rhob
    rhoc = params.rhoc
    rhop = params.rhop
    sc = 5.67e-8

    # Average mass concentration of the gas [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    # Gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    epb = (1 - ef0) * Ls / Lp

    Rep = rhog * np.abs(ug) * dp / mu
    Nup = (7 - 10 * afg + 5 * afg**2) * (1 + 0.7 * Rep**0.2 * Pr**0.33) + (1.33 - 2.4 * afg + 1.2 * afg**2) * Rep**0.7 * Pr**0.33
    hp = 6 * epb * kg * Nup / (phi * dp**2)

    rhob_s = rhob_b + rhob_c
    yc = rhob_c / rhob_s
    rhos = (yc / rhoc + (1 - yc) / rhob)**(-1)

    qp = (
        hp * (Tg - Tp) - 6 * es * sc * rhob_s / (rhos * ds) * (Tp**4 - Ts**4)
        + 4 / Db * epb * 1 / ((1 - ep) / (ep * epb) + (1 - ew) / ew + 1) * sc * (Tw**4 - Tp**4)
        - hps * (Tp - Ts)
    )

    rhopb = np.zeros(N)
    rhopb[0:Ni] = epb * rhop

    Cp = rhopb * cpp

    dtp_dt = np.zeros(N)
    dtp_dt[0:Ni] = (qp[0:Ni]) / Cp[0:Ni]

    return dtp_dt


def rhobb_rate(params, dx, mfg, rhob_b, rhob_g, Sb, v):
    """
    Calculate biomass mass concentration rate ∂ρ̅𝖻/∂t.
    """
    Ab = params.Ab
    ms_dot = params.ms_dot / 3600
    N = params.N
    N1 = params.N1
    Ni = params.Ni

    # Average mass concentration of the gas [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    # Gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    # Initialize array for biomass mass concentration rate ∂ρ̅𝖻/∂t
    # Note that ∂ρ̅𝖻/∂t is zero above bed, in the freeboard
    drhobb_dt = np.zeros(N)

    # Calculate ∂ρ̅𝖻/∂t below fuel inlet in bed
    drhobb_dt[0:N1 - 1] = -1 / dx[0:N1 - 1] * (-rhob_b[1:N1] * v[1:N1] + rhob_b[0:N1 - 1] * v[0:N1 - 1]) + Sb[0:N1 - 1]

    # Calculate ∂ρ̅𝖻/∂t at fuel inlet in bed
    vin = max(v[N1 - 1], ug[N1 - 1])
    rhobbin = ms_dot / (vin * Ab)
    drhobb_dt[N1 - 1] = -1 / dx[N1 - 1] * (-rhobbin * v[N1 - 1] + rhob_b[N1 - 1] * v[N1 - 1] - rhob_b[N1 - 2] * v[N1 - 2]) + Sb[N1 - 1]

    # Calculate ∂ρ̅𝖻/∂t above fuel inlet in bed
    drhobb_dt[N1:Ni] = -1 / dx[N1:Ni] * (rhob_b[N1:Ni] * v[N1:Ni] - rhob_b[N1 - 1:Ni - 1] * v[N1 - 1:Ni - 1]) + Sb[N1:Ni]

    return drhobb_dt


def rhobc_rate(params, dx, rhob_c, Sc, v):
    """
    here
    """
    N = params.N
    Ni = params.Ni

    drhobc_dt = np.zeros(N)
    drhobc_dt[0] = -1 / dx[0] * (-rhob_c[1] * v[1] + rhob_c[0] * v[0]) + Sc[0]
    drhobc_dt[1:Ni] = -1 / dx[1:Ni] * (-rhob_c[2:Ni + 1] * v[2:Ni + 1] + rhob_c[1:Ni] * v[1:Ni]) + Sc[1:Ni]

    return drhobc_dt


def rhobca_rate(params, dx, rhob_ca, Sca, v):
    """
    Calculate char accumulation rate.
    """
    N = params.N
    Ni = params.Ni

    drhobca_dt = np.zeros(N)
    drhobca_dt[0] = -1 / dx[0] * (-rhob_ca[1] * v[1] + rhob_ca[0] * v[0]) + Sca[0]
    drhobca_dt[1:Ni] = -1 / dx[1:Ni] * (-rhob_ca[2:Ni + 1] * v[2:Ni + 1] + rhob_ca[1:Ni] * v[1:Ni]) + Sca[1:Ni]

    return drhobca_dt


def v_rate(params, afg, dx, ds, ef, Lp, mfg, mu, rhog, rhob_b, rhob_c, rhob_g, Sb, Sc, umf, x, v):
    """
    Solid fuel velocity rate ∂v/∂t.
    """
    Ab = params.Ab
    Db = params.Db
    Ls = params.Ls
    Mgin = params.Mgin
    N = params.N
    Ni = params.Ni
    Pa = params.Pa
    SB = params.SB
    Tgin = params.Tgin
    cf = params.cf
    dp = params.dp
    e = params.e
    ef0 = params.ef0
    lb = params.lb
    ms_dot = params.ms_dot / 3600
    rhob = params.rhob
    rhoc = params.rhoc
    rhop = params.rhop

    fw = 0.25
    g = 9.81
    R = 8.314

    # Average mass concentration of the gas [kg/m³]
    rhob_gav = np.zeros(N)
    rhob_gav[0:N - 1] = 0.5 * (rhob_g[0:N - 1] + rhob_g[1:N])
    rhob_gav[N - 1] = rhob_g[N - 1]

    # Gas velocity along the reactor [m/s]
    ug = mfg / rhob_gav

    mfgin = SB * ms_dot / Ab
    Pin = (1 - ef0) * rhop * g * Ls + Pa
    rhog_in = Pin * Mgin / (R * Tgin) * 1e-3
    rhob_gin = rhog_in
    ugin = mfgin / rhob_gin

    Ugb = np.mean(np.append(afg[0:Ni] * ug[0:Ni], ugin))
    Ugb = max(ugin, Ugb)
    Dbu = 0.00853 * (1 + 27.2 * (Ugb - umf))**(1 / 3) * (1 + 6.84 * x)**1.21
    Vb = 1.285 * (Dbu / Db)**1.52 * Db

    ub = 12.51 * (Ugb - umf)**0.362 * (Dbu / Db)**0.52 * Db

    sfc = 2 * (3 / 2 * ds**2 * lb)**(2 / 3) / (ds * (ds + 2 * lb))

    Re_dc = rhog * np.abs(-ug - v) * ds / mu

    Cd = (
        24 / Re_dc * (1 + 8.1716 * Re_dc**(0.0964 + 0.5565 * sfc) * np.exp(-4.0655 * sfc))
        + Re_dc * 73.69 / (Re_dc + 5.378 * np.exp(6.2122 * sfc)) * np.exp(-5.0748 * sfc)
    )

    rhob_s = rhob_b + rhob_c
    yc = rhob_c / rhob_s
    rhos = (yc / rhoc + (1 - yc) / rhob)**(-1)

    epb = (1 - ef0) * Ls / Lp
    Yb = 1 / (1 + epb * rhos / rhob_s)
    afs = Yb * (1 - ef)

    rhopb = np.zeros(N)
    rhopb[0:Ni] = epb * rhop

    g0 = 1 / afg + 3 * ds * dp / (afg**2 * (dp + ds)) * (afs / ds + epb / dp)
    cs = 3 * np.pi * (1 + e) * (0.5 + cf * np.pi / 8) * (dp + ds)**2 / (rhop * dp**3 + rhos * ds**3) * afs * rhopb * g0

    Sp = Sb + Sc

    Spav = np.zeros(N)
    Spav[0:N - 1] = 0.5 * (Sp[0:N - 1] + Sp[1:N])
    Spav[N - 1] = Sp[N - 1]

    # Solid fuel velocity rate ∂v/∂t
    # Notice that above the bed, in the freeboard, is zero
    dvdt = np.zeros(N)

    # Solid fuel velocity rate ∂v/∂t in the bed
    dvdt[0:Ni] = (
        -1 / dx[0:Ni] * v[0:Ni] * (v[0:Ni] - v[1:Ni + 1])
        + g * (rhos[0:Ni] - rhog[0:Ni]) / rhos[0:Ni]
        + fw * rhop * Vb[0:Ni] / (dx[0:Ni] * rhos[0:Ni]) * (ub[0:Ni] - ub[1:Ni + 1])
        + (3 / 4) * (rhog[0:Ni] / rhos[0:Ni]) * (Cd[0:Ni] / ds[0:Ni]) * np.abs(-ug[0:Ni] - v[0:Ni]) * (-ug[0:Ni] - v[0:Ni])
        - cs[0:Ni] * v[0:Ni] * np.abs(v[0:Ni])
        + Spav[0:Ni] * v[0:Ni] / rhos[0:Ni]
    )

    return dvdt
