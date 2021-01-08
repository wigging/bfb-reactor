import numpy as np


def fluidization(params, Mg, Tg):
    """
    Fluidization properties related to the bed material.
    """
    Ab = params.Ab
    Db = params.Db
    Lmf = params.Lmf
    Lsi = params.Lsi
    Ni = params.Ni
    Pin = params.Pin
    SB = params.SB
    Tgin = params.Tgin
    dp = params.dp
    emf = params.emf
    ms_dot = params.ms_dot / 3600
    rhop = params.rhop

    R = 8.314
    g = 9.81

    Amu = np.array([27.758, 3.844, 23.811, 11.811, -36.826])
    Bmu = np.array([2.120, 4.0112, 5.3944, 4.9838, 4.290]) * 1e-1
    Cmu = np.array([-0.3280, -1.4303, -1.5411, -1.0851, -0.1620]) * 1e-4
    Tgm = np.mean(np.append(Tg[0:Ni], Tgin))
    Tgi = max(Tgin, Tgm)
    muin = (Amu[4] + Bmu[4] * Tgi + Cmu[4] * Tgi**2) * 1e-7

    Mgi = np.mean(Mg[0:Ni])
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

    return ef, Lp, umf
