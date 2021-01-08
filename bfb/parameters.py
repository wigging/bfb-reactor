import json
import numpy as np


class Parameters:

    def __init__(self, json_file):
        self.Db = None
        self.Dwo = None
        self.Lb = None
        self.Lu = None
        self.Lsi = None
        self.Mgin = None
        self.N1 = None
        self.N2 = None
        self.N3 = None
        self.P = None
        self.Pa = None
        self.SB = None
        self.Tgin = None
        self.db0 = None
        self.ef0 = None
        self.emf = None
        self.lb = None
        self.ms_dot = None
        self.rhob = None
        self.rhoc = None
        self.rhop = None
        self.wa = None
        self.wc = None
        self.xw = None

        # Assign parameters from JSON file. Commented lines with // are ignored.
        json_str = ''

        with open(json_file) as jfile:
            for line in jfile:
                if '//' not in line:
                    json_str += line

        json_data = json.loads(json_str)

        for key in json_data:
            setattr(self, key, json_data[key])

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
