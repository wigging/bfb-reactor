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

    # Inlet conditions
    Tsin = params['Tsin']
    rhogin = gas.rhog_inlet(params)
    mfgin, ugin = gas.mfg_ug_inlet(params, rhogin)
    mfsin, rhobbin = solid.mfs_rhobb_inlet(params, ugin)

    # Gas phase
    # afg = params['ef0']

    # Fluidization
    umf = gas.umf_bed(params, rhogin)

    # Solid fuel particle
    sfc = solid.sfc_fuel(params)

    # Initial values for gas and solid phase state variables
    Ts = np.full(N, Tsin)
    Tp = np.full(N, Tsin)
    mfg = np.full(N, mfgin)
    rhoab = np.full(N, 1e-8)
    rhobb = np.full(N, mfsin / ugin)
    rhocb = np.full(N, 1e-8)
    v = np.full(N, ugin)

    # Set count and tolerance for solver iterations
    count = 0
    divmg = 1
    torr = 1e-5

    while abs(divmg) > torr and count < 20:

        mfg_guess = np.mean(mfg)

        # Solid phase
        ya, yb, yc = solid.y_fracs(rhoab, rhobb, rhocb)
        cps = solid.cp_solid(Ts, yc)
        ds, rhos = solid.ds_rhos_fuel(params, ya, yb, yc)

        rhosb = rhoab + rhobb + rhocb
        afp, afs = solid.alpha_fracs(params, rhos, rhosb)

        # Gas phase
        ug = mfg / rhogin
        hgp = gas.hgp_conv(params, rhogin, ug)
        hgs = gas.hgs_conv(params, ds, rhogin, ug, v)
        Fb = gas.fb_prime(params, ug, ugin, umf, z)
        Sgs = 0.0
        Smgp = gas.betagp_momentum(params, afp, rhogin, ug)
        Smgs = gas.betags_momentum(params, ds, sfc, rhogin, ug, v)
        fg = gas.fg_factor(params, rhogin, ug)

        # Solid phase
        hps = solid.hps_coeff(params, afp, afs, cps, ds, rhogin, rhos, ug, yc, v)
        hwp = solid.hwp_conv(params, afp, rhogin)
        kb = kinetics.kb_rate(params, Ts)
        Kr = solid.kr_coeff(params, afp)
        Ms_res = solid.ms_res(params, Fb, rhogin, rhos)
        Smps = solid.betaps_momentum(params, afs, ds, mfsin, rhos, rhosb, v)
        Sa = kinetics.sa_gen(params, rhocb, Ts)
        Sb = kinetics.sb_gen(params, rhobb, Ts)
        Sc = kinetics.sc_gen(params, rhobb, rhocb, Ts)
        Sss = Sb + Sc + Sa
        Tp = solid.tp_inert(params, afp, afs, ds, hgp, hps, hwp, Kr, Tp, Ts)

        # Coefficients and A matrices
        aa, ba, ca = solid.rhoab_coeffs(dz, Sa, v)
        # ab, bb, cb = solid.rhobb_coeffs(params, dz, rhobbin, Sb, v)
        ab, bb, cb = solid.rhobb_coeffs2(params, dz, kb, mfsin, v)
        ac, bc, cc = solid.rhocb_coeffs(dz, Sc, v)
        # av, bv, cv = solid.v_coeffs(params, dz, rhos, Ms_res, Smgs, Smps, Sss, ug, v)
        av, bv, cv = solid.v_coeffs2(params, dz, rhos, Ms_res, Smgs, Smps, Sss, ug, v)
        ats, bts, cts = solid.ts_coeffs(params, afs, cps, ds, dz, hgs, hps, rhosb, Sb, Tp, Ts, v)
        # am, bm, cm, dm = gas.mfg_coeffs(params, afg, dz, fg, mfgin, rhogin, Sgs, Smgp, Smgs, ug, ugin, v)
        am, bm, cm, dm = gas.mfg_coeffs2(params, afs, dz, fg, mfgin, rhogin, Sgs, Smgp, Smgs, ug, ugin, v)

        Aa = diags([aa, -ba[1:]], offsets=[0, 1]).toarray()
        Ab = diags([ab, -bb[1:]], offsets=[0, 1]).toarray()
        Ac = diags([ac, -bc[1:]], offsets=[0, 1]).toarray()
        # Av = diags([av, -bv[0:N - 1]], offsets=[0, 1]).toarray()
        Av = diags([av, -bv[1:]], offsets=[0, 1]).toarray()
        Ats = diags([ats, -bts[0:N - 1]], offsets=[0, 1]).toarray()
        Am = diags([-am[1:N], bm, cm[0:N - 1]], offsets=[-1, 0, 1]).toarray()

        # Solve for state variables
        rhoab = np.linalg.solve(Aa, ca)
        rhobb = np.linalg.solve(Ab, cb)
        rhocb = np.linalg.solve(Ac, cc)
        v = np.linalg.solve(Av, cv)
        Ts = np.linalg.solve(Ats, cts)
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
        'rhoab': rhoab,
        'rhobb': rhobb,
        'rhobbin': rhobbin,
        'rhocb': rhocb,
        'ug': ug,
        'ugin': ugin,
        'v': v,
        'Tp': Tp,
        'Ts': Ts,
        'Tsin': Tsin
    }

    return results
