import numpy as np


class SolidPhase:

    def __init__(self, params):
        self._params = params

    def update_state(self, Tp, Ts, Tw, gas, rhob_b, rhob_c, rhob_ca, v):
        """
        Update state of the solid phase.
        """
        self.Tp = Tp
        self.Ts = Ts
        self.Tw = Tw
        self.rhob_b = rhob_b
        self.rhob_c = rhob_c
        self.rhob_ca = rhob_ca
        self.v = v
        self._calc_props()
        self._calc_hps(gas.Lp, gas.ef, gas.mu, gas.rhog, gas.ug)
        self._calc_qs(gas.Pr, gas.Tg, gas.kg, gas.mu, gas.rhog, gas.ug)

    def _calc_props(self):
        """
        Calculate various solid phase properties.
        """
        N = self._params.N
        db0 = self._params.db0
        lb = self._params.lb
        n1 = self._params.n1
        psi = self._params.psi
        rhob = self._params.rhob
        rhoc = self._params.rhoc

        Ts = self.Ts
        rhob_b = self.rhob_b
        rhob_c = self.rhob_c
        rhob_ca = self.rhob_ca

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

        # Assign calculated properties to class attributes
        self.Cs = Cs
        self.Xcr = Xcr
        self.cpb = cpb
        self.cpc = cpc
        self.cps = cps
        self.ds = ds
        self.rhob_s = rhob_s
        self.rhos = rhos
        self.sfc = sfc

    def _calc_hps(self, Lp, ef, mu, rhog, ug):
        """
        Calculate particle-particle heat transfer coefficient.
        """
        Db = self._params.Db
        Gp = self._params.Gp
        Gs = self._params.Gs
        Ls = self._params.Ls
        dp = self._params.dp
        cpp = self._params.cpp
        e = self._params.e
        ef0 = self._params.ef0
        gamp = self._params.gamp
        gams = self._params.gams
        kp = self._params.kp
        ks = self._params.ks
        rhop = self._params.rhop
        g = 9.81

        cps = self.cps
        ds = self.ds
        rhob_s = self.rhob_s
        rhos = self.rhos
        v = self.v

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

        self.hps = hps

    def _calc_qs(self, Pr, Tg, kg, mu, rhog, ug):
        """
        Calculate heat transfer due to gas flow and inert bed material. This
        method must be called after the `_calc_hps()` method.
        """
        es = self._params.es

        Tp = self.Tp
        Ts = self.Ts
        ds = self.ds
        hps = self.hps
        rhob_s = self.rhob_s
        rhos = self.rhos
        v = self.v
        sc = 5.67e-8

        Re_dc = abs(rhog) * abs(-ug - v) * ds / mu
        Nud = 2 + 0.6 * Re_dc**0.5 * Pr**0.33
        hs = Nud * kg / ds

        qs = (
            6 * hs * rhob_s / (rhos * ds) * (Tg - Ts)
            + 6 * es * sc * rhob_s / (rhos * ds) * (Tp**4 - Ts**4)
            + hps * (Tp - Ts)
        )

        self.qs = qs

    def rhobb_rate(self, dx, gas, kinetics):
        """
        Calculate biomass mass concentration rate ∂ρ̅𝖻/∂t.
        """
        Ab = self._params.Ab
        N = self._params.N
        Ni = self._params.Ni
        N1 = self._params.N1
        ms_dot = self._params.ms_dot / 3600

        rhob_b = self.rhob_b
        v = self.v

        # Gas phase and kinetics properties
        Sb = kinetics.Sb
        ug = gas.ug

        # Biomass mass concentration rate ∂ρ̅𝖻/∂t
        # ------------------------------------------------------------------------
        drhobb_dt = np.zeros(N)

        # Below fuel inlet in bed
        drhobb_dt[0:N1 - 1] = -1 / dx[0:N1 - 1] * (-rhob_b[1:N1] * v[1:N1] + rhob_b[0:N1 - 1] * v[0:N1 - 1]) + Sb[0:N1 - 1]

        # At fuel inlet in bed
        vin = max(v[N1 - 1], ug[N1 - 1])
        rhobbin = ms_dot / (vin * Ab)
        drhobb_dt[N1 - 1] = -1 / dx[N1 - 1] * (-rhobbin * v[N1 - 1] + rhob_b[N1 - 1] * v[N1 - 1] - rhob_b[N1 - 2] * v[N1 - 2]) + Sb[N1 - 1]

        # Above fuel inlet in bed
        drhobb_dt[N1:Ni] = -1 / dx[N1:Ni] * (rhob_b[N1:Ni] * v[N1:Ni] - rhob_b[N1 - 1:Ni - 1] * v[N1 - 1:Ni - 1]) + Sb[N1:Ni]

        return drhobb_dt

    def rhobc_rate(self, dx, kinetics):
        """
        here
        """
        N = self._params.N
        Ni = self._params.Ni

        Sc = kinetics.Sc
        rhob_c = self.rhob_c
        v = self.v

        drhobc_dt = np.zeros(N)
        drhobc_dt[0] = -1 / dx[0] * (-rhob_c[1] * v[1] + rhob_c[0] * v[0]) + Sc[0]
        drhobc_dt[1:Ni] = -1 / dx[1:Ni] * (-rhob_c[2:Ni + 1] * v[2:Ni + 1] + rhob_c[1:Ni] * v[1:Ni]) + Sc[1:Ni]

        return drhobc_dt

    def rhobca_rate(self, dx, kinetics):
        """
        Calculate char accumulation rate.
        """
        N = self._params.N
        Ni = self._params.Ni

        Sca = kinetics.Sca
        rhob_ca = self.rhob_ca
        v = self.v

        drhobca_dt = np.zeros(N)
        drhobca_dt[0] = -1 / dx[0] * (-rhob_ca[1] * v[1] + rhob_ca[0] * v[0]) + Sca[0]
        drhobca_dt[1:Ni] = -1 / dx[1:Ni] * (-rhob_ca[2:Ni + 1] * v[2:Ni + 1] + rhob_ca[1:Ni] * v[1:Ni]) + Sca[1:Ni]

        return drhobca_dt

    def tp_rate(self, gas):
        """
        here
        """
        Db = self._params.Db
        Ls = self._params.Ls
        N = self._params.N
        Ni = self._params.Ni
        cpp = self._params.cpp
        dp = self._params.dp
        ef0 = self._params.ef0
        ep = self._params.ep
        es = self._params.es
        ew = self._params.ew
        phi = self._params.phi
        rhop = self._params.rhop
        sc = 5.67e-8

        Tp = self.Tp
        Ts = self.Ts
        Tw = self.Tw
        ds = self.ds
        hps = self.hps
        rhob_s = self.rhob_s
        rhos = self.rhos

        # Gas phase properties
        Lp = gas.Lp
        Pr = gas.Pr
        Tg = gas.Tg
        afg = gas.afg
        kg = gas.kg
        mu = gas.mu
        rhog = gas.rhog
        ug = gas.ug

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
        rhopb[0:Ni] = epb * rhop

        Cp = rhopb * cpp

        dtp_dt = np.zeros(N)
        dtp_dt[0:Ni] = (qp[0:Ni]) / Cp[0:Ni]

        return dtp_dt

    def ts_rate(self, dx, kinetics):
        """
        Solid temperature rate ∂T𝗌/∂t.
        """
        N = self._params.N
        Ni = self._params.Ni
        N1 = self._params.N1
        Tsin = self._params.Tsin

        Cs = self.Cs
        Ts = self.Ts
        qs = self.qs
        v = self.v

        # Kinetics properties
        qss = kinetics.qss

        # Solid temperature rate ∂T𝗌/∂t
        # ------------------------------------------------------------------------
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

    def tw_rate(self, gas):
        """
        Calculate wall temperature rate.
        """

        # Parameters
        Db, Dwo, Dwi = self._params.Db, self._params.Dwo, self._params.Dwi
        Ls, N, Ni = self._params.Ls, self._params.N, self._params.Ni
        N1, Tam, Uha = self._params.N1, self._params.Tam, self._params.Uha
        dp, cpw, ef0 = self._params.dp, self._params.cpw, self._params.ef0
        ep, ew, kw = self._params.ep, self._params.ew, self._params.kw
        phi, rhow = self._params.phi, self._params.rhow
        sc = 5.67e-8

        # Solid phase properties
        Tp, Tw = self.Tp, self.Tw

        # Gas phase properties
        Lp, Pr, Tg = gas.Lp, gas.Pr, gas.Tg
        afg, kg, mu, rhog, ug = gas.afg, gas.kg, gas.mu, gas.rhog, gas.ug

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
        dtw_dt[0:Ni] = Qwb[0:Ni] / (mw * cpw)
        dtw_dt[Ni:N] = Qwf[Ni:N] / (mw * cpw)

        return dtw_dt

    def v_rate(self, dx, x, gas, kinetics):
        """
        Solid fuel velocity rate ∂v/∂t.
        """
        fw = 0.25
        g = 9.81

        # Parameters
        Db, Ls, N, Ni = self._params.Db, self._params.Ls, self._params.N, self._params.Ni
        cf, dp, e, ef0 = self._params.cf, self._params.dp, self._params.e, self._params.ef0
        lb, rhop, ugin = self._params.lb, self._params.rhop, self._params.ugin

        # Solid phase properties
        ds, rhob_s, rhos, v = self.ds, self.rhob_s, self.rhos, self.v

        # Gas phase properties
        Lp = gas.Lp
        afg = gas.afg
        ef = gas.ef
        mu = gas.mu
        rhog = gas.rhog
        ug = gas.ug
        umf = gas.umf

        Ugb = np.mean(np.append(afg[0:Ni] * ug[0:Ni], ugin))
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
        rhopb[0:Ni] = epb * rhop

        g0 = 1 / afg + 3 * ds * dp / (afg**2 * (dp + ds)) * (afs / ds + epb / dp)
        cs = 3 * np.pi * (1 + e) * (0.5 + cf * np.pi / 8) * (dp + ds)**2 / (rhop * dp**3 + rhos * ds**3) * afs * rhopb * g0

        Sp = kinetics.Sb + kinetics.Sc

        Spav = np.zeros(N)
        Spav[0:N - 1] = 0.5 * (Sp[0:N - 1] + Sp[1:N])
        Spav[N - 1] = Sp[N - 1]

        # Solid fuel velocity rate ∂v/∂t
        # ------------------------------------------------------------------------
        dvdt = np.zeros(N)

        # in the bed
        dvdt[0:Ni] = (
            -1 / dx[0:Ni] * v[0:Ni] * (v[0:Ni] - v[1:Ni + 1])
            + g * (rhos[0:Ni] - rhog[0:Ni]) / rhos[0:Ni]
            + fw * rhop * Vb[0:Ni] / (dx[0:Ni] * rhos[0:Ni]) * (ub[0:Ni] - ub[1:Ni + 1])
            + (3 / 4) * (rhog[0:Ni] / rhos[0:Ni]) * (Cd[0:Ni] / ds[0:Ni]) * np.abs(-ug[0:Ni] - v[0:Ni]) * (-ug[0:Ni] - v[0:Ni])
            - cs[0:Ni] * v[0:Ni] * np.abs(v[0:Ni])
            + Spav[0:Ni] * v[0:Ni] / rhos[0:Ni]
        )

        return dvdt
