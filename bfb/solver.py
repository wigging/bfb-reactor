import numpy as np
import logging
from scipy.integrate import solve_ivp

from dydt_system import dy_dt


def solver(params):
    """
    Solve system of ODEs.
    """

    # Parameters used for initial conditions
    mfg = params.mfg
    N = params.N
    rhobg = params.rhobg
    Tg = params.Tg
    Tp = params.Tp
    Ts = params.Ts
    ugin = params.ugin

    # Initial value arrays passed to ODE solver
    Ts_0 = np.full(N, Ts)
    Tg_0 = np.full(N, Tg)
    rhobb_0 = np.full(N, 1e-12)
    v_0 = np.full(N, ugin)
    mfg_0 = np.full(N, mfg)
    rhobg_0 = np.full(N, rhobg)
    rhobh2o_0 = np.full(N, rhobg)
    Tp_0 = np.full(N, Tp)
    rhobc_0 = np.zeros(N)
    rhobh2_0 = np.zeros(N)
    rhobch4_0 = np.zeros(N)
    rhobco_0 = np.zeros(N)
    rhobco2_0 = np.zeros(N)
    rhobt_0 = np.zeros(N)
    rhobca_0 = np.zeros(N)
    Tw_0 = np.full(N, Tg)

    # Initial state for solver
    y0 = np.concatenate(
        (Ts_0, Tg_0, rhobb_0, v_0, mfg_0, rhobg_0, rhobh2o_0, Tp_0, rhobc_0,
         rhobh2_0, rhobch4_0, rhobco_0, rhobco2_0, rhobt_0, rhobca_0, Tw_0)
    )

    # Solve system of ODEs using SciPy ODE solver
    tspan = (0, params.tf)
    sol = solve_ivp(dy_dt, tspan, y0, method='LSODA', args=(params,))

    # Log information
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    log = (
        f't0      {sol.t[0]}\n'
        f'tf      {sol.t[-1]}\n'
        f'N       {N}\n'
        f'len t   {len(sol.t)}\n'
        f'y shape {sol.y.shape}'
    )

    logging.info(log)

    # Results for plotting and analysis
    results = {
        'N': N,
        't': sol.t,
        'Ts': sol.y[0:N],
        'Tg': sol.y[N:2 * N],
        'rhob_b': sol.y[2 * N:3 * N],
        'v': sol.y[3 * N:4 * N],
        'mfg': sol.y[4 * N:5 * N],
        'rhob_g': sol.y[5 * N:6 * N],
        'rhob_h2o': sol.y[6 * N:7 * N],
        'Tp': sol.y[7 * N:8 * N],
        'rhob_c': sol.y[8 * N:9 * N],
        'rhob_h2': sol.y[9 * N:10 * N],
        'rhob_ch4': sol.y[10 * N:11 * N],
        'rhob_co': sol.y[11 * N:12 * N],
        'rhob_co2': sol.y[12 * N:13 * N],
        'rhob_t': sol.y[13 * N:14 * N],
        'rhob_ca': sol.y[14 * N:15 * N],
        'Tw': sol.y[15 * N:]
    }

    return results
