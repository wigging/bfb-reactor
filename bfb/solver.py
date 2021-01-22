import numpy as np
from scipy.integrate import solve_ivp

from gas_phase import GasPhase
from solid_phase import SolidPhase
from kinetics import Kinetics
from dydt_system import dydt


def solver(params):
    """
    Setup initial conditions and run SciPy ODE solver.
    """
    N = params.N
    Ni = params.Ni

    # Setup gas phase, solid phase, and kinetics objects
    gas = GasPhase(params)
    solid = SolidPhase(params)
    kinetics = Kinetics(params)

    # Initial conditions
    Tg0 = np.full(N, params.Tg)
    Tp0 = np.zeros(N)
    Tp0[0:Ni] = params.Tp
    Ts0 = np.full(N, params.Ts)
    Tw0 = np.full(N, params.Tg)
    rhobb0 = np.full(N, 1e-12)
    rhobc0 = np.zeros(N)
    rhobca0 = np.zeros(N)
    v0 = np.full(N, params.ugin)
    rhobh20 = np.zeros(N)
    rhobh2o0 = np.full(N, params.rhobg)
    rhobch40 = np.zeros(N)
    rhobco0 = np.zeros(N)
    rhobco20 = np.zeros(N)
    rhobt0 = np.zeros(N)
    mfg0 = np.full(N, params.mfg)

    # Initial state for solver
    y0 = np.concatenate((
        Tg0, Tp0, Ts0, Tw0, rhobb0, rhobc0, rhobca0, v0, rhobh20, rhobh2o0,
        rhobch40, rhobco0, rhobco20, rhobt0, mfg0))

    # Solve system of ODEs using SciPy ODE solver
    tspan = (0, params.tf)
    args = (params, gas, solid, kinetics)
    sol = solve_ivp(dydt, tspan, y0, method='Radau', args=args)

    # Print solver info
    print(f'\n{" Solver Info ":-^60}\n')
    print(f'{"message:":10} {sol.message}')
    print(f'{"success:":10} {sol.success}')
    print(f'{"nfev:":10} {sol.nfev}')
    print(f'{"njev:":10} {sol.njev}')
    print(f'{"nlu:":10} {sol.nlu}')

    # Print results info
    print(f'\n{" Results Info ":-^60}\n')
    print(
        f't0      {sol.t[0]}\n'
        f'tf      {sol.t[-1]}\n'
        f'N       {N}\n'
        f'len t   {len(sol.t)}\n'
        f'y shape {sol.y.shape}'
    )

    # Return results from solver
    results = {
        't': sol.t,
        'Ts': sol.y[2 * N:3 * N],
        'rhob_b': sol.y[4 * N:5 * N],
        'v': sol.y[7 * N:8 * N]
    }

    return results
