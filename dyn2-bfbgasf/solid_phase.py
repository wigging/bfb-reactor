import numpy as np


# >>>
# FIXME: these variables should be calculated, assumes that N = 100
Cs = np.full(100, 1.8e-09)
Xcr = np.full(100, 0.5)
qs = np.full(100, 0.00086252)
qs[-25:] = -0.00032331
qss = np.full(100, -3.6442e-19)
v = np.full(100, 0.34607)
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


def ts_rate(params, dx, N, Np, Ts):
    """
    Solid temperature rate ∂T𝗌/∂t.
    """
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
