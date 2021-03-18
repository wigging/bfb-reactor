import gas_phase as gas
import solid_phase as solid


def dydt(t, y, params, dx, N, Np):
    """
    Right-hand side of the system of ODEs.
    """

    # variables from the solver
    mfg = y[0:N]

    # solid properties
    ds = solid.ds_fuel(params)
    sfc = solid.sfc_fuel(params, ds)

    # gas properties
    DP = gas.calc_dP(dx, N, Np)
    rhob_gav = gas.calc_rhobgav(N)

    # gas mass flux terms
    Cmf, Smgg, SmgV = gas.mfg_terms(params, ds, dx, mfg, N, Np, rhob_gav, sfc)

    # differential equations
    dmfg_dt = gas.mfg_rate(params, Cmf, dx, DP, mfg, rhob_gav, N, Np, Smgg, SmgV)

    # system of equations
    dy_dt = dmfg_dt

    return dy_dt
