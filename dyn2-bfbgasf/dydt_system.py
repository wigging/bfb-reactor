import numpy as np

import gas_phase as gas
import solid_phase as solid


def dydt(t, y, params, dx):
    """
    Right-hand side of the system of ODEs.
    """
    N = params['N']
    Np = params['Np']

    # variables from the solver
    Ts = y[0:N]
    mfg = y[N:2 * N]

    # solid properties
    ds = solid.calc_ds(params)
    sfc = solid.calc_sfc(params, ds)

    # gas properties
    DP = gas.calc_dP(dx, N, Np)
    rhob_gav = gas.calc_rhobgav(N)

    # gas mass flux terms
    Cmf, Smgg, SmgV = gas.mfg_terms(params, ds, dx, mfg, N, Np, rhob_gav, sfc)

    # differential equations
    dTs_dt = solid.ts_rate(params, dx, N, Np, Ts)
    dmfg_dt = gas.mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, N, Np, Smgg, SmgV)

    # system of equations
    dy_dt = np.concatenate((dTs_dt, dmfg_dt))

    return dy_dt
