import numpy as np

import gas_phase as gas
import solid_phase as solid
import kinetics


def dydt(t, y, params, dx, x):
    """
    Right-hand side of the system of ODEs.
    """
    N = params['N']

    # variables from the solver
    Ts = y[0:N]
    Tg = y[N:2 * N]
    rhobb = y[2 * N:3 * N]
    v = y[3 * N:4 * N]
    mfg = y[4 * N:5 * N]
    rhobg = y[5 * N:6 * N]

    # kinetics
    Sb = kinetics.calc_sb(rhobb, Ts)

    # solid properties
    ds = solid.calc_ds(params)
    sfc = solid.calc_sfc(params, ds)

    # gas properties
    Mg, Pr, cpg, cpgm, kg, mu, xg = gas.calc_mix_props(rhobg, Tg)
    DP = gas.calc_dP(params, dx, Mg, rhobg, Tg)
    rhob_gav = gas.calc_rhobgav(N, rhobg)

    # gas mass flux terms
    Cmf, Smgg, SmgV = gas.mfg_terms(params, ds, dx, mfg, mu, rhobg, rhob_gav, sfc, v)

    # differential equations
    dTs_dt = solid.ts_rate(params, dx, Ts, v)
    dTg_dt = gas.tg_rate(params, cpg, ds, dx, kg, mfg, mu, Pr, rhobg, rhob_gav, Tg, Ts, v)
    drhobb_dt = solid.rhobb_rate(params, dx, mfg, rhobb, rhob_gav, Sb, v)
    dv_dt = solid.v_rate(params, ds, dx, mfg, mu, rhobg, rhob_gav, Sb, v, x)
    dmfg_dt = gas.mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, Smgg, SmgV)
    drhobg_dt = gas.rhobg_rate(params, dx, mfg)

    # system of equations
    dy_dt = np.concatenate((dTs_dt, dTg_dt, drhobb_dt, dv_dt, dmfg_dt, drhobg_dt))

    return dy_dt
