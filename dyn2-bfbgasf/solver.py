import numpy as np
from scipy.integrate import solve_ivp

from grid import grid
from dydt_system import dydt


def solver(params):
    """
    Solver function.
    """
    N = params['N']

    # one-dimensional grid steps and points
    dx, x = grid(params)

    # initial conditions
    Ts0 = np.full(N, 300)
    Tg0 = np.full(N, 1100)
    mfg0 = np.full(N, 0.2)

    # solve system of ODEs using SciPy ODE solver
    y0 = np.concatenate((Ts0, Tg0, mfg0))
    tspan = (0, params['tf'])
    args = (params, dx)
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
        'mfg': sol.y[2 * N:3 * N]
    }

    return results
