import numpy as np

import gas_phase as gas
import solid_phase as solid
import kinetics


def dydt(t, y, params, dx, x):
    """
    Right-hand side of the system of ODEs.
    """
    N = params['N']

    # State variables from solver
    state = {
        'Ts': y[0:N],
        'Tg': y[N:2 * N],
        'rhob_b': y[2 * N:3 * N],
        'v': y[3 * N:4 * N],
        'mfg': y[4 * N:5 * N],
        'rhob_g': y[5 * N:6 * N],
        'rhob_h2o': y[6 * N:7 * N],
        'Tp': y[7 * N:8 * N],
        'rhob_c': y[8 * N:9 * N],
        'rhob_h2': y[9 * N:10 * N],
        'rhob_ch4': y[10 * N:11 * N],
        'rhob_co': y[11 * N:12 * N],
        'rhob_co2': y[12 * N:13 * N],
        'rhob_t': y[13 * N:14 * N],
        'rhob_ca': y[14 * N:15 * N],
        'Tw': y[15 * N:16 * N]
    }

    # Gas phase properties
    Mg, Pr, cpg, kg, mu, xg = gas.calc_mix_props(state)
    Lp, ef, umf = gas.calc_bedexp(params, Mg=Mg, Tg=state['Tg'])
    afg, DP, P, rhob_gav, rhog, ug = gas.calc_props(params, state, dx, ef, Mg)

    # Solid phase properties
    Cs, Xcr, cps, ds, rhob_s, rhos, sfc = solid.calc_props(params, state)
    hps = solid.calc_hps(params, state, cps, ds, ef, Lp, mu, rhob_s, rhog, rhos, ug)
    qs = solid.calc_qs(params, state, ds, hps, kg, mu, Pr, rhob_s, rhog, rhos, ug)

    # Kinetics
    Sb, Sc, Sca, Sh2, Sh2o, Sch4, Sco, Sco2, Sg, St = kinetics.calc_sgen(params, state, P, Xcr, xg)
    qgs = kinetics.calc_qgs(state)
    qss = kinetics.calc_qss(params, state, P, Sb, Xcr, xg)

    # Gas mass flux terms
    Cmf, Smgg, SmgV = gas.mfg_terms(params, state, afg, ds, dx, ef, Lp, mu, rhob_gav, rhob_s, rhog, rhos, sfc, Sg, ug)

    # Differential equations
    dTs_dt = solid.ts_rate(params, state, Cs, dx, qs, qss)
    dTg_dt = gas.tg_rate(params, state, afg, cpg, ds, dx, kg, Lp, mu, Pr, qgs, rhob_s, rhog, rhos, ug)
    drhobb_dt = solid.rhobb_rate(params, state, dx, Sb, ug)
    dv_dt = solid.v_rate(params, state, afg, ds, dx, ef, Lp, mu, rhob_s, rhog, rhos, Sb, Sc, ug, umf, x)
    dmfg_dt = gas.mfg_rate(params, state, Cmf, dx, DP, rhob_gav, Smgg, SmgV, ug)
    drhobg_dt = gas.rhobg_rate(params, state, dx, Sg)
    drhobh2o_dt = gas.rhobx_rate(params, state, dx, Sh2o, x='rhob_h2o', rhob_xin=params['rhob_gin'])
    dTp_dt = solid.tp_rate(params, state, afg, ds, hps, kg, Lp, mu, Pr, rhob_s, rhog, rhos, ug)
    drhobc_dt = solid.rhobc_rate(params, state, dx, Sc)
    drhobh2_dt = gas.rhobx_rate(params, state, dx, Sh2, x='rhob_h2', rhob_xin=0)
    drhobch4_dt = gas.rhobx_rate(params, state, dx, Sch4, x='rhob_ch4', rhob_xin=0)
    drhobco_dt = gas.rhobx_rate(params, state, dx, Sco, x='rhob_co', rhob_xin=0)
    drhobco2_dt = gas.rhobx_rate(params, state, dx, Sco2, x='rhob_co2', rhob_xin=0)
    drhobt_dt = gas.rhobx_rate(params, state, dx, St, x='rhob_t', rhob_xin=0)
    drhobca_dt = solid.rhobca_rate(params, state, dx, Sca)
    dTw_dt = solid.tw_rate(params, state, afg, kg, Lp, mu, Pr, rhog, ug)

    # System of equations
    dy_dt = np.concatenate((
        dTs_dt, dTg_dt, drhobb_dt, dv_dt, dmfg_dt, drhobg_dt, drhobh2o_dt,
        dTp_dt, drhobc_dt, drhobh2_dt, drhobch4_dt, drhobco_dt, drhobco2_dt,
        drhobt_dt, drhobca_dt, dTw_dt))

    return dy_dt
