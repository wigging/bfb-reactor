import numpy as np

import gas_phase as gas
import solid_phase as solid


def dydt(t, y, params, dx):
    """
    Right-hand side of the system of ODEs.
    """
    N = params['N']

    # variables from the solver
    Ts = y[0:N]
    Tg = y[N:2 * N]
    mfg = y[2 * N:3 * N]

    # solid properties
    ds = solid.calc_ds(params)
    sfc = solid.calc_sfc(params, ds)

    # gas properties
    DP = gas.calc_dP(params, dx, Tg)
    rhob_gav = gas.calc_rhobgav(N)
    Mg, Pr, cpg, cpgm, kg, mu, xg = gas.calc_mix_props(Tg)

    # gas mass flux terms
    Cmf, Smgg, SmgV = gas.mfg_terms(params, ds, dx, mfg, rhob_gav, sfc)

    # differential equations
    dTs_dt = solid.ts_rate(params, dx, Ts)
    dTg_dt = gas.tg_rate(params, ds, dx, mfg, rhob_gav, Tg, Ts)
    dmfg_dt = gas.mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, Smgg, SmgV)

    # system of equations
    dy_dt = np.concatenate((dTs_dt, dTg_dt, dmfg_dt))

    return dy_dt
