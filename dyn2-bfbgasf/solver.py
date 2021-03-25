import numpy as np
from scipy.integrate import solve_ivp

from grid import grid
from dydt_system import dydt


def solver(params):
    """
    Solver function.
    """
    N = params['N']
    Np = params['Np']

    # one-dimensional grid steps and points
    dx, x = grid(params, Lp=0.2347)

    # initial conditions
    Ts0 = np.full(N, 300)
    Tg0 = np.full(N, 1100)
    rhobb0 = np.full(N, 1e-12)
    v0 = np.full(N, params['ugin'])
    mfg0 = np.full(N, 0.2)
    rhobg0 = np.full(N, 0.15)
    rhobh2o0 = np.full(N, 0.15)
    Tp0 = np.zeros(N)
    Tp0[0:Np] = 1100
    rhobc0 = np.full(N, 0)
    rhobh20 = np.full(N, 1e-12)
    rhobch40 = np.full(N, 1e-12)
    rhobco0 = np.full(N, 1e-12)
    rhobco20 = np.full(N, 1e-12)
    rhobt0 = np.full(N, 1e-12)
    rhobca0 = np.full(N, 1e-12)
    Tw0 = np.full(N, 1100)

    # solve system of ODEs using SciPy ODE solver
    y0 = np.concatenate((
        Ts0, Tg0, rhobb0, v0, mfg0, rhobg0, rhobh2o0, Tp0, rhobc0, rhobh20,
        rhobch40, rhobco0, rhobco20, rhobt0, rhobca0, Tw0))
    tspan = (0, params['tf'])
    args = (params, dx, x)
    sol = solve_ivp(dydt, tspan, y0, method='Radau', rtol=1e-6, args=args)

    # print solver information
    print(f'\n{" Solver Info ":-^60}\n')
    print(f'{"message:":10} {sol.message}')
    print(f'{"success:":10} {sol.success}')
    print(f'{"nfev:":10} {sol.nfev}')
    print(f'{"njev:":10} {sol.njev}')
    print(f'{"nlu:":10} {sol.nlu}')

    # return results dictionary
    results = {
        'x': x,
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
        'Tw': sol.y[15 * N:16 * N]
    }

    return results
