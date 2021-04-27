import numpy as np
from scipy.integrate import solve_ivp

from gas_phase import calc_bedexp, calc_bedexp2
from grid import grid
from dydt_system import dydt


def solver(params):
    """
    Solver function.
    """
    N = params['N']
    Np = params['Np']

    # Fluidization conditions
    # here

    # One-dimensional grid steps and points
    Hf = calc_bedexp(params)
    Lp = calc_bedexp2(params)
    breakpoint()
    dx, x = grid(params, Lp=0.2347)

    # Initial conditions arrays
    Ts0 = np.full(N, params['Ts0'])
    Tg0 = np.full(N, params['Tg0'])
    rhobb0 = np.full(N, params['rhobb0'])
    v0 = np.full(N, params['ugin'])
    mfg0 = np.full(N, params['mfg0'])
    rhobg0 = np.full(N, params['rhobg0'])
    rhobh2o0 = np.full(N, params['rhobh2o0'])
    Tp0 = np.zeros(N)
    Tp0[0:Np] = params['Tp0']
    rhobc0 = np.full(N, params['rhobc0'])
    rhobh20 = np.full(N, params['rhobh20'])
    rhobch40 = np.full(N, params['rhobch40'])
    rhobco0 = np.full(N, params['rhobco0'])
    rhobco20 = np.full(N, params['rhobco20'])
    rhobt0 = np.full(N, params['rhobt0'])
    rhobca0 = np.full(N, params['rhobca0'])
    Tw0 = np.full(N, params['Tw0'])

    # Solve system of ODEs using SciPy ODE solver
    y0 = np.concatenate((
        Ts0, Tg0, rhobb0, v0, mfg0, rhobg0, rhobh2o0, Tp0, rhobc0, rhobh20,
        rhobch40, rhobco0, rhobco20, rhobt0, rhobca0, Tw0))
    tspan = (0, params['tf'])
    args = (params, dx, x)
    sol = solve_ivp(dydt, tspan, y0, method=params['method'], rtol=params['rtol'], args=args)

    # Print solver and results information
    print(
        f'\n{" Solver Info ":-^60}\n'
        f'message {sol.message}\n'
        f'success {sol.success}\n'
        f'nfev    {sol.nfev}\n'
        f'njev    {sol.njev}\n'
        f'nlu     {sol.nlu}'
        f'\n{" Results Info ":-^60}\n'
        f't0      {sol.t[0]}\n'
        f'tf      {sol.t[-1]}\n'
        f'len t   {len(sol.t)}\n'
        f'y shape {sol.y.shape}'
    )

    # Return results dictionary
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
