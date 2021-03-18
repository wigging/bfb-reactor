import numpy as np
from scipy.integrate import solve_ivp

from grid import grid
from dydt_system import dydt


def solver(params):
    """
    Solver function.
    """
    N1 = params['N1']
    N2 = params['N2']
    N3 = params['N3']

    # total grid points (N) and grid points to bed top (Np)
    N = N1 + N2 + N3
    Np = N1 + N2

    # one-dimensional grid steps and points
    dx, x = grid(params)

    # initial conditions
    mfg0 = np.full(N, 0.2)

    # solve system of ODEs using SciPy ODE solver
    y0 = mfg0
    tspan = (0, params['tf'])
    args = (params, dx, N, Np)
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
        'mfg': sol.y
    }

    return results
