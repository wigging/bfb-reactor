import numpy as np


class Parameters:

    def __init__(self, params):

        self.Db = params['Db']
        self.Dwo = params['Dwo']
        self.Gp = params['Gp']
        self.Gs = params['Gs']
        self.L = params['L']
        self.Lb = params['Lb']
        self.Lu = params['Lu']
        self.Lsi = params['Lsi']
        self.Mgin = params['Mgin']
        self.N1 = params['N1']
        self.N2 = params['N2']
        self.N3 = params['N3']
        self.P = params['P']
        self.Pa = params['Pa']
        self.SB = params['SB']
        self.Tg = params['Tg']
        self.Tp = params['Tp']
        self.Ts = params['Ts']
        self.Tam = params['Tam']
        self.Tgin = params['Tgin']
        self.Tsin = params['Tsin']
        self.Uha = params['Uha']

        self.cf = params['cf']
        self.cpp = params['cpp']
        self.cpw = params['cpw']
        self.db0 = params['db0']
        self.dp = params['dp']
        self.e = params['e']
        self.ef0 = params['ef0']
        self.emf = params['emf']
        self.ep = params['ep']
        self.es = params['es']
        self.ew = params['ew']
        self.gamp = params['gamp']
        self.gams = params['gams']
        self.kp = params['kp']
        self.ks = params['ks']
        self.kw = params['kw']
        self.lb = params['lb']
        self.ms_dot = params['ms_dot']
        self.mfg = params['mfg']
        self.n1 = params['n1']
        self.phi = params['phi']
        self.rhob = params['rhob']
        self.rhoc = params['rhoc']
        self.rhop = params['rhop']
        self.rhow = params['rhow']
        self.rhobg = params['rhobg']
        self.tf = params['tf']
        self.wa = params['wa']
        self.wc = params['wc']
        self.wH2O = params['wH2O']
        self.xw = params['xw']

    @property
    def ugin(self):
        R = 8.314
        Ab = (np.pi / 4) * (self.Db**2)
        mfgin = self.SB * (self.ms_dot / 3600) / Ab
        rhogin = self.P * self.Mgin / (R * self.Tgin) * 1e-3
        ugin = mfgin / rhogin
        return ugin

    @property
    def mfgin(self):
        Ab = (np.pi / 4) * (self.Db**2)
        mfgin = self.SB * (self.ms_dot / 3600) / Ab
        return mfgin

    @property
    def N(self):
        N = self.N1 + self.N2 + self.N3
        return N

    @property
    def Ni(self):
        Ni = self.N1 + self.N2
        return Ni

    @property
    def Ab(self):
        Ab = (np.pi / 4) * (self.Db**2)
        return Ab

    @property
    def db(self):
        db = 3 * self.db0 * self.lb / (2 * self.lb + self.db0)
        return db

    @property
    def Dwi(self):
        Dwi = self.Dwo - 2 * self.xw
        return Dwi

    @property
    def Lf0(self):
        Lf0 = self.Lb - self.Lu
        return Lf0

    @property
    def Lmf(self):
        Ls = self.Lsi - self.Lu
        Lmf = (1 - self.ef0) / (1 - self.emf) * Ls
        return Lmf

    @property
    def Ls(self):
        Ls = self.Lsi - self.Lu
        return Ls

    @property
    def Pin(self):
        g = 9.81
        Ls = self.Lsi - self.Lu
        Pin = (1 - self.ef0) * self.rhop * g * Ls + self.Pa
        return Pin

    @property
    def psi(self):
        psi = self.rhoc / (self.rhob * (self.wc + self.wa))
        return psi
