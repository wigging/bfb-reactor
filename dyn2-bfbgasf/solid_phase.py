import numpy as np


def ds_fuel(params):
    """
    Average diameter of the solid fuel particle.
    """

    # >>> FIXME
    yc = np.full(100, 1e-8)
    Xcr = np.full(100, 0.5)
    # <<<

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


def sfc_fuel(params, ds):
    """
    Sphericity or effective shape factor of the solid fuel particle.
    """
    lb = params['lb']

    # sphericity of solid fuel particle [-]
    sfc = 2 * ((3 / 2) * ds**2 * lb)**(2 / 3) / (ds * (ds + 2 * lb))
    return sfc
