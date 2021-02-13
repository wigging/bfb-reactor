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

    # Inlet properties
    rhogin = gas.rhog_inlet(params)
    mfgin, ugin = gas.mfg_ug_inlet(params, rhogin)
    mfsin, rhobbin = solid.mfs_rhobb_inlet(params, ugin)

    # Gas phase calculations
    afg = params['ef0']

    # Fluidization calculations
    umf = gas.umf_bed(params, rhogin)

    # Solid phase calculations
    ds = solid.ds_fuel(params)
    rhos = solid.rhos_density(params)
    sfc = solid.sfc_fuel(params)

    # Initial values for gas phase state variables
    mfg = np.full(N, mfgin)
    rhobb = np.full(N, mfsin / ugin)
    rhobc = np.full(N, 1e-8)
    v = np.full(N, ugin)

    # Set count and tolerance for solver iterations
    count = 0
    divmg = 1
    torr = 1e-5

    # Solve for gas phase and solid phase state variables
    while abs(divmg) > torr and count < 20:

        mfg_guess = np.mean(mfg)

        # Solid inlet
        vin = v[-1]
        rhobbin = mfsin / vin

        # Gas phase variables ------------------------------------------------

        ug = mfg / rhogin
        Fb = gas.fb_prime(params, afg, ug, ugin, umf, z)
        Mg_res = gas.mg_prime(params, afg, dz, rhogin)
        Sgs = 0.0
        Smgp = gas.betagp_momentum(params, rhogin, ug)
        Smgs = gas.betags_momentum(params, ds, sfc, rhogin, ug, v)
        fg = gas.fg_factor(params, rhogin, ug)

        # Solid phase variables ----------------------------------------------

        Ms_res = solid.ms_res(params, Fb, rhogin, rhos)
        Smps = solid.betaps_momentum(params, afg, ds, mfsin, rhos, rhobb, rhobc, v)
        Sb = kinetics.sb_gen(params, rhobb)
        Sc = kinetics.sc_gen(params, rhobb)
        Sss = Sb + Sc

        # Matrix -------------------------------------------------------------

        ab, bb, cb = solid.rhobb_coeffs(params, dz, rhobbin, Sb, v)
        ac, bc, cc = solid.rhobc_coeffs(dz, Sc, v)
        av, bv, cv = solid.v_coeffs(params, dz, rhos, Ms_res, Smgs, Smps, Sss, ug, v)
        am, bm, cm, dm = gas.mfg_coeffs(params, afg, dz, fg, mfgin, rhogin, Mg_res, Sgs, Smgp, Smgs, ug, ugin, v)

        Ab = diags([ab, -bb[0:N - 1]], offsets=[0, 1]).toarray()
        Ac = diags([ac, -bc[0:N - 1]], offsets=[0, 1]).toarray()
        Av = diags([av, -bv[0:N - 1]], offsets=[0, 1]).toarray()
        Am = diags([-am[1:N], bm, cm[0:N - 1]], offsets=[-1, 0, 1]).toarray()

        # Solve --------------------------------------------------------------

        rhobb = np.linalg.solve(Ab, cb)
        rhobc = np.linalg.solve(Ac, cc)
        v = np.linalg.solve(Av, cv)
        mfg = np.linalg.solve(Am, dm)

        rhobb = np.maximum(rhobb, 0)

        # Update count and guess ---------------------------------------------

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
