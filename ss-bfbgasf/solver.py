import numpy as np
from scipy.sparse import diags

import gas_phase as gas
import solid_phase as solid
import kinetics


def _grid(params):
    """
    One-dimensional grid along reactor height [m].
    """
    L = params['L']
    N = params['N']

    z = np.linspace(0, L, N + 1)
    dz = z[1] - z[0] / 2

    return z, dz


def solver(params):
    """
    Solve system of equations for given parameters.
    """
    N = params['N']
    z, dz = _grid(params)

    # Inlet properties for gas phase
    rhogin = gas.rhog_inlet(params)
    mfgin, ugin = gas.mfg_ug_inlet(params, rhogin)

    # Inlet properties for solid phase
    mfsin, rhobbin = solid.mfs_rhobb_inlet(params, ugin)

    # Gas phase calculations
    afg = params['ef0']

    # Fluidization calculations
    umf = gas.umf_bed(params, rhogin)

    # Solid phase calculations
    ds = solid.ds_fuel(params)
    rhos = solid.rhos_density(params)
    sfc = solid.sfc_fuel(params)

    # Set initial values for gas phase state variables
    mfg = np.full(N, mfgin)

    # Set initial values for solid phase state variables
    rhobb = np.full(N, 1e-8)
    rhobc = np.full(N, 1e-8)
    v = np.full(N, ugin)

    # Set count and tolerance for solver iterations
    count = 0
    divmg = 1
    torr = 1e-5

    # Solve for gas phase and solid phase state variables
    while abs(divmg) > torr and count < 20:

        mfg_guess = np.mean(mfg)

        # Gas phase calculations
        ug = mfg / rhogin
        Fb = gas.fb_prime(params, afg, ug, ugin, umf, z)
        Mg_res = gas.mg_prime(params, afg, dz, rhogin)
        Sgs = 0.0
        Smgp = gas.betagp_momentum(params, rhogin, ug)
        Smgs = gas.betags_momentum(params, ds, sfc, rhogin, ug, v)
        fg = gas.fg_factor(params, rhogin, ug)

        # Solid phase calculations
        Ms_res = solid.ms_res(params, Fb, rhogin, rhos)
        Smps = solid.betaps_momentum(params, afg, ds, mfsin, rhos, v)
        Sss = 0.0
        Sb = kinetics.sb_gen(params, rhobb)
        Sc = kinetics.sc_gen(params, rhobb)

        # Biomass mass concentration - rhobb
        ab, bb, cb = solid.rhobb_coeffs(params, dz, rhobbin, Sb, v)
        Ab = diags([ab, -bb[1:]], offsets=[0, 1]).toarray()
        rhobb = np.linalg.solve(Ab, cb)
        rhobb = np.maximum(rhobb, 0)

        # Char mass concentration - rhobc
        ac, bc, cc = solid.rhobc_coeffs(dz, Sc, v)
        Ac = diags([ac, -bc[1:]], offsets=[0, 1]).toarray()
        rhobc = np.linalg.solve(Ac, cc)

        # Solid fuel velocity - v
        av, bv, cv = solid.v_coeffs(params, dz, rhos, Ms_res, Smgs, Smps, Sss, ug, v)
        Av = diags([av, -bv[1:]], offsets=[0, 1]).toarray()
        v = np.linalg.solve(Av, cv)

        # Gas mass flux - mfg
        am, bm, cm, dm = gas.mfg_coeffs(params, afg, dz, fg, mfgin, rhogin, Mg_res, Sgs, Smgp, Smgs, ug, ugin, v)
        Am = diags([-am[1:], bm, cm[1:]], offsets=[-1, 0, 1]).toarray()
        mfg = np.linalg.solve(Am, dm)

        # Update count and guess
        count = count + 1
        divmg = np.mean(mfg) - mfg_guess

    # Print solver information
    print(f'divmg = {divmg:.4g}')
    print(f'count = {count}')

    # Return results for plotting
    results = {
        'z': z,
        'mfg': mfg,
        'mfgin': mfgin,
        'rhobb': rhobb,
        'rhobbin': rhobbin,
        'rhobc': rhobc,
        'ug': ug,
        'ugin': ugin,
        'v': v
    }

    return results
