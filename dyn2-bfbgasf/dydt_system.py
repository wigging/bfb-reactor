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
    rhob_g = y[5 * N:6 * N]
    rhob_h2o = y[6 * N:7 * N]
    Tp = y[7 * N:8 * N]

    # solid properties
    ds = solid.calc_ds(params)
    sfc = solid.calc_sfc(params, ds)

    # gas properties
    Mg, Pr, cpg, cpgm, kg, mu, xg = gas.calc_mix_props(rhob_g, rhob_h2o, Tg)
    DP = gas.calc_dP(params, dx, Mg, rhob_g, Tg)
    rhob_gav = gas.calc_rhobgav(N, rhob_g)

    # kinetics
    Sb = kinetics.calc_sb(rhobb, Ts)
    Sh2o = kinetics.calc_h2o(params, Mg, rhob_g, rhob_h2o, Sb, Tg, Ts, xg)

    # gas mass flux terms
    Cmf, Smgg, SmgV = gas.mfg_terms(params, ds, dx, mfg, mu, rhob_g, rhob_gav, sfc, v)

    # differential equations
    dTs_dt = solid.ts_rate(params, dx, Ts, v)
    dTg_dt = gas.tg_rate(params, cpg, ds, dx, kg, mfg, mu, Pr, rhob_g, rhob_gav, Tg, Ts, v)
    drhobb_dt = solid.rhobb_rate(params, dx, mfg, rhobb, rhob_gav, Sb, v)
    dv_dt = solid.v_rate(params, ds, dx, mfg, mu, rhob_g, rhob_gav, Sb, v, x)
    dmfg_dt = gas.mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, Smgg, SmgV)
    drhobg_dt = gas.rhobg_rate(params, dx, mfg)
    drhobh2o_dt = gas.rhobh2o_rate(params, dx, mfg, rhob_g, rhob_h2o, Sh2o)
    dTp_dt = solid.tp_rate(params, ds, kg, mfg, mu, Pr, rhob_g, rhob_gav, Tg, Tp, Ts,)

    # system of equations
    dy_dt = np.concatenate((dTs_dt, dTg_dt, drhobb_dt, dv_dt, dmfg_dt, drhobg_dt, drhobh2o_dt, dTp_dt))

    return dy_dt
