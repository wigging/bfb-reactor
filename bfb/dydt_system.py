import numpy as np
from grid import grid


def dydt(t, y, params, gas, solid, kinetics):
    """
    Right-hand side of the system of ODEs.
    """
    N = params.N

    # Variables from the solver
    Tg = y[0:N]
    Tp = y[N:2 * N]
    Ts = y[2 * N:3 * N]
    Tw = y[3 * N:4 * N]
    rhob_b = y[4 * N:5 * N]
    rhob_c = y[5 * N:6 * N]
    rhob_ca = y[6 * N:7 * N]
    v = y[7 * N:8 * N]
    rhob_h2 = y[8 * N:9 * N]
    rhob_h2o = y[9 * N:10 * N]
    rhob_ch4 = y[10 * N:11 * N]
    rhob_co = y[11 * N:12 * N]
    rhob_co2 = y[12 * N:13 * N]
    rhob_t = y[13 * N:14 * N]
    mfg = y[14 * N:15 * N]

    # Update gas phase, solid phase, and kinetics properties
    gas.update_state(Tg, mfg, rhob_h2, rhob_h2o, rhob_ch4, rhob_co, rhob_co2, rhob_t)
    solid.update_state(Tp, Ts, Tw, gas, rhob_b, rhob_c, rhob_ca, v)
    kinetics.update_state(gas, solid)

    # One-dimensional grid
    dx, x = grid(params, gas.Lp)

    # Rate equations
    dTg_dt = gas.tg_rate(dx, kinetics, solid)
    dTp_dt = solid.tp_rate(gas)
    dTs_dt = solid.ts_rate(dx, kinetics)
    dTw_dt = solid.tw_rate(gas)
    drhobb_dt = solid.rhobb_rate(dx, gas, kinetics)
    drhobc_dt = solid.rhobc_rate(dx, kinetics)
    drhobca_dt = solid.rhobca_rate(dx, kinetics)
    dv_dt = solid.v_rate(dx, x, gas, kinetics)
    drhobh2_dt = gas.rhobh2_rate(dx, kinetics)
    drhobh2o_dt = gas.rhobh2o_rate(dx, kinetics)
    drhobch4_dt = gas.rhobch4_rate(dx, kinetics)
    drhobco_dt = gas.rhobco_rate(dx, kinetics)
    drhobco2_dt = gas.rhobco2_rate(dx, kinetics)
    drhobt_dt = gas.rhobt_rate(dx, kinetics)
    dmfg_dt = gas.mfg_rate(dx, kinetics, solid)

    # Combine rate equations into system of ODEs for solver
    dy_dt = np.concatenate((
        dTg_dt, dTp_dt, dTs_dt, dTw_dt, drhobb_dt, drhobc_dt, drhobca_dt,
        dv_dt, drhobh2_dt, drhobh2o_dt, drhobch4_dt, drhobco_dt, drhobco2_dt,
        drhobt_dt, dmfg_dt))

    return dy_dt
