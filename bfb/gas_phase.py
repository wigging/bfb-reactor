import numpy as np

M_CH4 = 16
M_CO = 28
M_CO2 = 44
M_H2 = 2
M_H2O = 18


class GasPhase:

    def __init__(self, params):
        self._params = params

    def update_state(self, Tg, mfg, rhob_h2, rhob_h2o, rhob_ch4, rhob_co, rhob_co2, rhob_t):
        """
        Update state of the gas phase.
        """
        self.Tg = Tg
        self.mfg = mfg
        self.rhob_h2 = rhob_h2
        self.rhob_h2o = rhob_h2o
        self.rhob_ch4 = rhob_ch4
        self.rhob_co = rhob_co
        self.rhob_co2 = rhob_co2
        self.rhob_t = rhob_t
        self.rhob_g = rhob_h2 + rhob_h2o + rhob_ch4 + rhob_co + rhob_co2 + rhob_t
        self._calc_rhobgav_ug()
        self._calc_yfracs()
        self._calc_mix_props()
        self._calc_fluidization()
        self._calc_afg_rhog_p()

    def _calc_rhobgav_ug(self):
        """
        Calculate average mass concentration of the gas and gas velocity along
        the reactor.
        """
        N = self._params.N

        # Average gas mass concentration [kg/m³]
        rhob_gav = np.zeros(N)
        rhob_gav[0:N - 1] = 0.5 * (self.rhob_g[0:N - 1] + self.rhob_g[1:N])
        rhob_gav[N - 1] = self.rhob_g[N - 1]

        # Gas velocity along the reactor [m/s]
        ug = self.mfg / rhob_gav

        # Assign to class attributes
        self.rhob_gav = rhob_gav
        self.ug = ug

    def _calc_yfracs(self):
        """
        Calculate gas species mass fractions.
        """
        yH2 = self.rhob_h2 / self.rhob_g
        yH2O = self.rhob_h2o / self.rhob_g
        yCH4 = self.rhob_ch4 / self.rhob_g
        yCO = self.rhob_co / self.rhob_g
        yCO2 = self.rhob_co2 / self.rhob_g
        yt = self.rhob_t / self.rhob_g
        yg = np.column_stack((yH2, yCH4, yCO, yCO2, yH2O))

        self.yH2 = yH2
        self.yH2O = yH2O
        self.yCH4 = yCH4
        self.yCO = yCO
        self.yCO2 = yCO2
        self.yt = yt
        self.yg = yg

    def _calc_mix_props(self):
        """
        Calculate gas mixture properties along the reactor.
        """
        N = self._params.N
        Tg = self.Tg

        Amu = np.array([27.758, 3.844, 23.811, 11.811, -36.826])
        Bmu = np.array([2.120, 4.0112, 5.3944, 4.9838, 4.290]) * 1e-1
        Cmu = np.array([-0.3280, -1.4303, -1.5411, -1.0851, -0.1620]) * 1e-4

        Acp = np.array([25.399, 34.942, 29.556, 27.437, 33.933])
        Bcp = np.array([20.178, -39.957, -6.5807, 42.315, -8.4186]) * 1e-3
        Ccp = np.array([-3.8549, 19.184, 2.0130, -1.9555, 2.9906]) * 1e-5
        Dcp = np.array([3.188, -15.303, -1.2227, 0.39968, -1.7825]) * 1e-8
        Ecp = np.array([-8.7585, 39.321, 2.2617, -0.29872, 3.6934]) * 1e-12

        Ak = np.array([3.951, -0.935, 0.158, -1.200, 0.053]) * 1e-2
        Bk = np.array([4.5918, 1.4028, 0.82511, 1.0208, 0.47093]) * 1e-4
        Ck = np.array([-6.4933, 3.3180, 1.9081, -2.2403, 4.9551]) * 1e-8

        # Calculations
        M_g = np.array([M_H2, M_CH4, M_CO, M_CO2, M_H2O])

        n = len(M_g)
        mug = np.zeros((N, n))
        cpgg = np.zeros((N, n))
        kgg = np.zeros((N, n))

        for j in range(n):
            mug[:, j] = (Amu[j] + Bmu[j] * Tg + Cmu[j] * Tg**2) * 1e-7
            cpgg[:, j] = Acp[j] + Bcp[j] * Tg + Ccp[j] * Tg**2 + Dcp[j] * Tg**3 + Ecp[j] * Tg**4
            kgg[:, j] = Ak[j] + Bk[j] * Tg + Ck[j] * Tg**2

        Mg = np.zeros(N)
        mu = np.zeros(N)
        xg = np.zeros((N, n))
        cpgm = np.zeros(N)
        kg = np.zeros(N)

        for i in range(N):
            xgm = (self.yg[i] / M_g) / np.sum(self.yg[i] / M_g)
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

        cpt = -100 + 4.40 * Tg - 1.57e-3 * Tg**2
        cpgg = cpgm / Mg * 1e3
        cpg = self.yt * cpt + (1 - self.yt) * cpgg
        Pr = cpg * mu / kg

        self.Mg = Mg
        self.Pr = Pr
        self.cpg = cpg
        self.cpgm = cpgm
        self.kg = kg
        self.mu = mu
        self.xg = xg

    def _calc_fluidization(self):
        """
        Calculate fluidization properties. This method must be called after
        the `_calc_mix_props()` method.
        """
        Ab = self._params.Ab
        Db = self._params.Db
        Lmf = self._params.Lmf
        Lsi = self._params.Lsi
        Ni = self._params.Ni
        Pin = self._params.Pin
        SB = self._params.SB
        Tgin = self._params.Tgin
        dp = self._params.dp
        emf = self._params.emf
        ms_dot = self._params.ms_dot / 3600
        rhop = self._params.rhop
        R = 8.314
        g = 9.81

        Amu = np.array([27.758, 3.844, 23.811, 11.811, -36.826])
        Bmu = np.array([2.120, 4.0112, 5.3944, 4.9838, 4.290]) * 1e-1
        Cmu = np.array([-0.3280, -1.4303, -1.5411, -1.0851, -0.1620]) * 1e-4

        Tgm = np.mean(np.append(self.Tg[0:Ni], Tgin))
        Tgi = max(Tgin, Tgm)
        muin = (Amu[4] + Bmu[4] * Tgi + Cmu[4] * Tgi**2) * 1e-7

        Mgi = np.mean(self.Mg[0:Ni])
        rhogi = Pin * Mgi / (R * Tgi) * 1e-3

        Ar = dp**3 * rhogi * (rhop - rhogi) * g / muin**2
        Rem = -33.67 + (33.67**2 + 0.0408 * Ar)**0.5
        umf = Rem * muin / (rhogi * dp)
        Umsr = (np.exp(-0.5405 * Lsi / Db) * (4.294e3 / Ar + 1.1) + 3.676e2 * Ar**(-1.5) + 1)

        mfgin = SB * ms_dot / Ab
        Ugin = mfgin / rhogi

        Drbs = 1
        Rrb = (1 - 0.103 * (Umsr * umf - umf)**(-0.362) * Drbs)**(-1)
        Rrs = (1 - 0.305 * (Ugin - umf)**(-0.362) * Db**0.48)**(-1)
        Dbr = 5.64e-4 / (Db * Lmf) * (1 + 27.2 * (Ugin - umf))**(1 / 3) * ((1 + 6.84 * Lmf)**2.21 - 1)

        if Dbr < Drbs:
            Re = (1 - 0.103 * (Ugin - umf)**(-0.362) * Dbr)**(-1)
        else:
            Re = Rrb * Rrs

        De = Re - 1
        if np.isnan(De) or De <= 0:
            De = 0.05

        ef = 1 - (1 - emf) / (De + 1)
        Lp = (De + 1) * Lmf

        self.Lp = Lp
        self.ef = ef
        self.umf = umf

    def _calc_afg_rhog_p(self):
        """
        Calculate volume fraction and density of the gas along the reactor.
        This method must be called after the `_calc_fluidization()` method.
        """
        N = self._params.N
        Ni = self._params.Ni
        ef = self.ef
        R = 8.314

        # Volume fraction of gas in bed and freeboard [-]
        afg = np.ones(N)
        afg[0:Ni] = ef

        # Density of gas along reactor axis [kg/m³]
        rhog = self.rhob_g / afg

        # Pressure of gas along reactor axis [Pa]
        P = R * rhog * self.Tg / self.Mg * 1e3

        # Assign to class attributes
        self.afg = afg
        self.rhog = rhog
        self.P = P

    def mfg_rate(self, dx, kinetics, solid):
        """
        Gas mass flux rate ∂ṁfg/∂t.
        """
        Ab = self._params.Ab
        Db = self._params.Db
        Ls = self._params.Ls
        Mgin = self._params.Mgin
        N = self._params.N
        Ni = self._params.Ni
        N1 = self._params.N1
        Pa = self._params.Pa
        SB = self._params.SB
        Tgin = self._params.Tgin
        dp = self._params.dp
        ef0 = self._params.ef0
        ms_dot = self._params.ms_dot / 3600
        phi = self._params.phi
        rhop = self._params.rhop

        Sg = kinetics.Sg
        afg = self.afg
        ef = self.ef
        mfg = self.mfg
        mu = self.mu
        rhob_gav = self.rhob_gav
        rhog = self.rhog
        ug = self.ug

        R = 8.314
        g = 9.81

        # Solid phase properties
        ds = solid.ds
        rhob_s = solid.rhob_s
        rhos = solid.rhos
        sfc = solid.sfc
        v = solid.v

        # Calculations
        # ------------------------------------------------------------------------
        Pin = (1 - ef0) * rhop * g * Ls + Pa
        rhog_in = Pin * Mgin / (R * Tgin) * 1e-3
        rhob_gin = rhog_in

        DP = -afg[0:N - 1] / dx[0:N - 1] * (self.P[1:N] - self.P[0:N - 1])

        epb = (1 - ef0) * Ls / self.Lp

        vin = max(v[N1 - 1], ug[N1 - 1])
        rhobbin = ms_dot / (vin * Ab)

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
        # --------------------------------------------------------------------
        dmfgdt = np.zeros(N)

        # at gas inlet, bottom of reactor
        mfgin = SB * ms_dot / Ab
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

    def rhobh2_rate(self, dx, kinetics):
        """
        here
        """
        N = self._params.N
        ugin = self._params.ugin

        Sh2 = kinetics.Sh2
        mfg = self.mfg
        yH2 = self.yH2

        rhob_h2in = 0

        drhobh2_dt = np.zeros(N)
        drhobh2_dt[0] = -(yH2[0] * mfg[0] - rhob_h2in * ugin) / dx[0] + Sh2[0]
        drhobh2_dt[1:N] = -(yH2[1:N] * mfg[1:N] - yH2[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sh2[1:N]

        return drhobh2_dt

    def rhobh2o_rate(self, dx, kinetics):
        """
        here
        """
        Mgin = self._params.Mgin
        N = self._params.N
        Pin = self._params.Pin
        Tgin = self._params.Tgin
        ugin = self._params.ugin
        R = 8.314

        Sh2o = kinetics.Sh2o
        mfg = self.mfg
        yH2O = self.yH2O

        rhog_in = Pin * Mgin / (R * Tgin) * 1e-3
        rhob_h2oin = rhog_in

        drhobh2o_dt = np.zeros(N)
        drhobh2o_dt[0] = -(yH2O[0] * mfg[0] - rhob_h2oin * ugin) / dx[0] + Sh2o[0]
        drhobh2o_dt[1:N] = -(yH2O[1:N] * mfg[1:N] - yH2O[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sh2o[1:N]

        return drhobh2o_dt

    def rhobch4_rate(self, dx, kinetics):
        """
        here
        """
        N = self._params.N
        ugin = self._params.ugin

        Sch4 = kinetics.Sch4
        mfg = self.mfg
        yCH4 = self.yCH4
        rhob_ch4in = 0

        drhobch4_dt = np.zeros(N)
        drhobch4_dt[0] = -(yCH4[0] * mfg[0] - rhob_ch4in * ugin) / dx[0] + Sch4[0]
        drhobch4_dt[1:N] = -(yCH4[1:N] * mfg[1:N] - yCH4[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sch4[1:N]

        return drhobch4_dt

    def rhobco_rate(self, dx, kinetics):
        """
        here
        """
        N = self._params.N
        ugin = self._params.ugin

        Sco = kinetics.Sco
        mfg = self.mfg
        yCO = self.yCO
        rhob_coin = 0

        drhobco_dt = np.zeros(N)
        drhobco_dt[0] = -(yCO[0] * mfg[0] - rhob_coin * ugin) / dx[0] + Sco[0]
        drhobco_dt[1:N] = -(yCO[1:N] * mfg[1:N] - yCO[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sco[1:N]

        return drhobco_dt

    def rhobco2_rate(self, dx, kinetics):
        """
        here
        """
        N = self._params.N
        ugin = self._params.ugin

        Sco2 = kinetics.Sco2
        mfg = self.mfg
        yCO2 = self.yCO2
        rhob_co2in = 0

        drhobco2_dt = np.zeros(N)
        drhobco2_dt[0] = -(yCO2[0] * mfg[0] - rhob_co2in * ugin) / dx[0] + Sco2[0]
        drhobco2_dt[1:N] = -(yCO2[1:N] * mfg[1:N] - yCO2[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + Sco2[1:N]

        return drhobco2_dt

    def rhobt_rate(self, dx, kinetics):
        """
        here
        """
        N = self._params.N
        ugin = self._params.ugin

        St = kinetics.St
        mfg = self.mfg
        yt = self.yt
        rhob_tin = 0

        drhobt_dt = np.zeros(N)
        drhobt_dt[0] = -(yt[0] * mfg[0] - rhob_tin * ugin) / dx[0] + St[0]
        drhobt_dt[1:N] = -(yt[1:N] * mfg[1:N] - yt[0:N - 1] * mfg[0:N - 1]) / dx[1:N] + St[1:N]

        return drhobt_dt

    def tg_rate(self, dx, kinetics, solid):
        """
        Gas temperature rate representing the ∂T𝗀/∂t equation.
        """
        Db = self._params.Db
        Dwi = self._params.Dwi
        Dwo = self._params.Dwo
        Ls = self._params.Ls
        N = self._params.N
        Ni = self._params.Ni
        Tgin = self._params.Tgin
        dp = self._params.dp
        ef0 = self._params.ef0
        kw = self._params.kw
        phi = self._params.phi

        # Gas phase properties
        Lp = self.Lp
        Pr = self.Pr
        Tg = self.Tg
        afg = self.afg
        cpg = self.cpg
        kg = self.kg
        mu = self.mu
        rhob_g = self.rhob_g
        rhog = self.rhog
        ug = self.ug

        # Solid phase and kinetics properties
        Tp = solid.Tp
        Ts = solid.Ts
        Tw = solid.Tw
        ds = solid.ds
        qgs = kinetics.qgs
        rhob_s = solid.rhob_s
        rhos = solid.rhos
        v = solid.v

        # - - -

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

        # Gas temperature rate ∂T𝗀/∂t
        # ------------------------------------------------------------------------
        dtgdt = np.zeros(N)

        dtgdt[0] = -ug[0] / (dx[0]) * (Tg[0] - Tgin) + (-qgs[0] + qg[0]) / Cg[0]

        dtgdt[1:Ni] = -ug[1:Ni] / dx[1:Ni] * (Tg[1:Ni] - Tg[0:Ni - 1]) + (-qgs[1:Ni] + qg[1:Ni]) / Cg[1:Ni]

        dtgdt[Ni:N] = (
            -ug[Ni:N] / dx[Ni:N] * (Tg[Ni:N] - Tg[Ni - 1:N - 1])
            - (qgs[Ni:N] - 4 / Db * Uhf[Ni:N] * (Tw[Ni:N] - Tg[Ni:N])) / Cg[Ni:N]
        )

        return dtgdt
