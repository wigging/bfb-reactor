import numpy as np

import gas
import solid
import kinetics
from fluidization import fluidization
from grid import grid


def dy_dt(t, y, params):
    """
    Create system of ODEs for solver.
    """
    N = params.N

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
    Tw = y[15 * N:]

    # Collect variables into tuples to pass to functions
    rhobz = (
        rhob_b, rhob_c, rhob_ca, rhob_ch4, rhob_co, rhob_co2, rhob_g,
        rhob_h2, rhob_h2o, rhob_t
    )

    Tz = (Tg, Tp, Ts, Tw)

    # Gas mixture properties along the reactor
    cpg, cpgm, kg, Mg, mu, Pr, xg = gas.mix_props(params, rhobz, Tg)

    # Fluidization properties
    ef, Lp, umf = fluidization(params, Mg, Tg)

    # Gas volume fraction
    afg = gas.volume_frac(params, ef)

    # Gas density and pressure
    rhog, P = gas.density_press(afg, rhob_g, Mg, Tg)

    # One-dimensional grid
    dx, x = grid(params, Lp)

    # Char conversion and particle diameter
    Xcr = solid.char_conv(params, rhob_c, rhob_ca)
    ds = solid.particle_diam(params, rhob_b, rhob_c, Xcr)

    # Heat capacities and heat transfer coefficient
    cpb, cpc, cps, Cs = solid.heat_cap(params, rhob_b, rhob_c, Ts)
    hps = solid.heat_coeff(params, cps, ds, ef, Lp, mfg, mu, rhog, rhob_b, rhob_c, rhob_g, v)

    # Gas and solid phase mass generation rates
    Sb, Sc, Sca = kinetics.solid_gen(params, P, rhob_b, rhob_c, rhob_h2, Tg, Ts, Xcr, xg)
    Sch4, Sco, Sco2, Sh2, Sh2o, St, Sg = kinetics.gas_gen(params, P, rhobz, Sb, Tg, Ts, Xcr, xg)

    # Solid temperature rate ∂T𝗌/∂t
    dTs_dt = solid.ts_rate(params, Cs, dx, ds, kg, hps, mfg, mu, P, Pr, rhog, rhobz, Sb, Tz, v, Xcr, xg)

    # Gas temperature rate ∂T𝗀/∂t
    dTg_dt = gas.tg_rate(params, cpg, ds, dx, ef, kg, Lp, mfg, mu, Pr, rhobz, Tz, v)

    # Biomass mass concentration rate ∂ρ̅𝖻/∂t
    drhobb_dt = solid.rhobb_rate(params, dx, mfg, rhob_b, rhob_g, Sb, v)

    # Solid fuel velocity rate ∂v/∂t
    dv_dt = solid.v_rate(params, afg, dx, ds, ef, Lp, mfg, mu, rhog, rhob_b, rhob_c, rhob_g, Sb, Sc, umf, x, v)

    # Gas mass flux rate ∂ṁfg/∂t
    dmfg_dt = gas.mfg_rate(params, ds, dx, ef, Lp, mfg, Mg, mu, rhobz, Sg, Tg, v)

    # Bulk gas mass concentration rate ∂ρ̅𝗀/∂t
    drhobg_dt = gas.rhobg_rate(params, dx, mfg, Sg)

    # H₂O mass concentration rate ∂ρ̅H₂O/∂t
    ugin = gas.inlet_flow(params)
    drhobh2o_dt = gas.rhobh2o_rate(params, dx, mfg, rhob_g, rhob_h2o, Sh2o, ugin)

    # Bed particle temperature rate ∂T𝗉/∂t
    dTp_dt = solid.tp_rate(params, afg, ds, hps, kg, Lp, mfg, mu, Pr, rhog, rhob_b, rhob_c, rhob_g, Tg, Tp, Ts, Tw)

    # Char mass concentration rate ∂ρ̅𝖼/∂t
    drhobc_dt = solid.rhobc_rate(params, dx, rhob_c, Sc, v)

    # H₂ mass concentration rate ∂ρ̅H₂/∂t
    drhobh2_dt = gas.rhobh2_rate(params, dx, mfg, rhob_g, rhob_h2, Sh2, ugin)

    # CH₄ mass concentration rate ∂ρ̅CH₄/∂t
    drhobch4_dt = gas.rhobch4_rate(params, dx, mfg, rhob_ch4, rhob_g, Sch4, ugin)

    # CO mass concentration rate ∂ρ̅CO/∂t
    drhobco_dt = gas.rhobco_rate(params, dx, mfg, rhob_co, rhob_g, Sco, ugin)

    # CO₂ mass concentration rate ∂ρ̅CO₂/∂t
    drhobco2_dt = gas.rhobco2_rate(params, dx, mfg, rhob_co2, rhob_g, Sco2, ugin)

    # Tar mass concentration rate ∂ρ̅𝗍/∂t
    drhobt_dt = gas.rhobt_rate(params, dx, mfg, rhob_t, rhob_g, St, ugin)

    # Char accumulation rate ∂ρ̅𝖼𝖺/∂t
    drhobca_dt = solid.rhobca_rate(params, dx, rhob_ca, Sca, v)

    # Wall temperature rate ∂T𝗐/∂t
    dTw_dt = solid.tw_rate(params, afg, kg, mfg, mu, Lp, Pr, rhog, rhob_g, Tg, Tp, Tw)

    # Combine rate equations into system of ODEs
    dydt = np.concatenate((
        dTs_dt, dTg_dt, drhobb_dt, dv_dt, dmfg_dt, drhobg_dt, drhobh2o_dt,
        dTp_dt, drhobc_dt, drhobh2_dt, drhobch4_dt, drhobco_dt, drhobco2_dt,
        drhobt_dt, drhobca_dt, dTw_dt
    ))

    return dydt
