import numpy as np

import gas_phase as gas
import solid_phase as solid
import kinetics


def dydt(t, y, params, dx, x):
    """
    Right-hand side of the system of ODEs.
    """
    N = params['N']

    # Variables from the solver
    Ts = y[0:N]
    Tg = y[N:2 * N]
    rhob_b = y[2 * N:3 * N]
    v = y[3 * N:4 * N]
    mfg = y[4 * N:5 * N]
    rhob_g = y[5 * N:6 * N]
    rhob_h2o = y[6 * N:7 * N]
    Tp = y[7 * N:8 * N]
    rhob_c = y[8 * N:9 * N]
    rhob_h2 = y[9 * N:10 * N]
    rhob_ch4 = y[10 * N:11 * N]
    rhob_co = y[11 * N:12 * N]
    rhob_co2 = y[12 * N:13 * N]
    rhob_t = y[13 * N:14 * N]
    rhob_ca = y[14 * N:15 * N]
    Tw = y[15 * N:16 * N]

    # Gas phase properties
    Mg, Pr, cpg, cpgm, kg, mu, xg = gas.calc_mix_props(rhob_ch4, rhob_co, rhob_co2, rhob_g, rhob_h2, rhob_h2o, rhob_t, Tg)
    Lp, ef, umf = gas.calc_fluidization(params, Mg, Tg)
    P, DP = gas.calc_dP(params, dx, ef, Mg, rhob_g, Tg)
    rhob_gav = gas.calc_rhobgav(N, rhob_g)

    # Solid phase properties
    Cs, Xcr, cpb, cpc, cps, ds, rhob_s, rhos, sfc = solid.calc_props(params, rhob_b, rhob_c, rhob_ca, Ts)
    hps = solid.calc_hps(params, cps, ds, ef, Lp, mfg, mu, rhob_g, rhob_gav, rhob_s, rhos, v)
    qs = solid.calc_qs(params, ds, ef, hps, kg, mfg, mu, Pr, rhob_g, rhob_gav, rhob_s, rhos, Tg, Tp, Ts, v)

    # Kinetics
    Sb, Sc, Sca, Sh2, Sh2o, Sch4, Sco, Sco2, Sg, St = kinetics.calc_sgen(
        params, P, Ts, Tg, Xcr, rhob_b, rhob_c, rhob_ch4, rhob_co, rhob_co2, rhob_h2, rhob_h2o, rhob_t, xg)
    qgs = kinetics.calc_qgs(rhob_ch4, rhob_co, rhob_co2, rhob_h2, rhob_h2o, Tg)
    qss = kinetics.calc_qss(params, P, rhob_c, rhob_h2, Sb, Tg, Ts, Xcr, xg)

    # Gas mass flux terms
    Cmf, Smgg, SmgV = gas.mfg_terms(params, ds, dx, ef, Lp, mfg, mu, rhob_g, rhob_gav, rhob_s, rhos, sfc, Sg, v)

    # Differential equations
    dTs_dt = solid.ts_rate(params, Cs, dx, qs, qss, Ts, v)
    dTg_dt = gas.tg_rate(params, cpg, ds, dx, ef, kg, Lp, mfg, mu, Pr, qgs, rhob_g, rhob_gav, rhob_s, rhos, Tg, Tp, Ts, Tw, v)
    drhobb_dt = solid.rhobb_rate(params, dx, mfg, rhob_b, rhob_gav, Sb, v)
    dv_dt = solid.v_rate(params, ds, dx, ef, Lp, mfg, mu, rhob_g, rhob_gav, rhob_s, rhos, Sb, Sc, umf, v, x)
    dmfg_dt = gas.mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, Smgg, SmgV)
    drhobg_dt = gas.rhobg_rate(params, dx, mfg, Sg)
    drhobh2o_dt = gas.rhobh2o_rate(params, dx, mfg, rhob_g, rhob_h2o, Sh2o)
    dTp_dt = solid.tp_rate(params, ds, ef, hps, kg, Lp, mfg, mu, Pr, rhob_g, rhob_gav, rhob_s, rhos, Tg, Tp, Ts, Tw)
    drhobc_dt = solid.rhobc_rate(params, dx, rhob_c, Sc, v)
    drhobh2_dt = gas.rhobh2_rate(params, dx, mfg, rhob_g, rhob_h2, Sh2)
    drhobch4_dt = gas.rhobch4_rate(params, dx, mfg, rhob_g, rhob_ch4, Sch4)
    drhobco_dt = gas.rhobco_rate(params, dx, mfg, rhob_g, rhob_co, Sco)
    drhobco2_dt = gas.rhobco2_rate(params, dx, mfg, rhob_g, rhob_co2, Sco2)
    drhobt_dt = gas.rhobt_rate(params, dx, mfg, rhob_g, rhob_t, St)
    drhobca_dt = solid.rhobca_rate(params, dx, rhob_ca, Sca, v)
    dTw_dt = solid.tw_rate(params, ef, kg, Lp, mfg, mu, Pr, rhob_g, rhob_gav, Tg, Tp, Tw)

    # System of equations
    dy_dt = np.concatenate((
        dTs_dt, dTg_dt, drhobb_dt, dv_dt, dmfg_dt, drhobg_dt, drhobh2o_dt,
        dTp_dt, drhobc_dt, drhobh2_dt, drhobch4_dt, drhobco_dt, drhobco2_dt,
        drhobt_dt, drhobca_dt, dTw_dt))

    return dy_dt
