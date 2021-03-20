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
    rhobb = y[2 * N:3 * N]
    mfg = y[3 * N:4 * N]

    # solid properties
    ds = solid.calc_ds(params)
    sfc = solid.calc_sfc(params, ds)

    # gas properties
    Mg, Pr, cpg, cpgm, kg, mu, xg = gas.calc_mix_props(Tg)
    DP = gas.calc_dP(params, dx, Mg, Tg)
    rhob_gav = gas.calc_rhobgav(N)

    # gas mass flux terms
    Cmf, Smgg, SmgV = gas.mfg_terms(params, ds, dx, mfg, mu, rhob_gav, sfc)

    # differential equations
    dTs_dt = solid.ts_rate(params, dx, Ts)
    dTg_dt = gas.tg_rate(params, cpg, ds, dx, kg, mfg, mu, Pr, rhob_gav, Tg, Ts)
    drhobb_dt = solid.rhobb_rate(params, dx, mfg, rhobb, rhob_gav)
    dmfg_dt = gas.mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, Smgg, SmgV)

    # system of equations
    dy_dt = np.concatenate((dTs_dt, dTg_dt, drhobb_dt, dmfg_dt))

    return dy_dt
