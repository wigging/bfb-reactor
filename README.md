# One-dimensional BFB gasifier model

:warning: This project is a work in progress.

## Usage

Execute the `main.py` file to run the model.

```bash
$ python main.py
```

The current model will run for several minutes and eventually output the warnings shown below. I eventually terminated the program after 26 minutes because the solver had not converged to a solution.

```
solid_phase.py:139: RuntimeWarning: invalid value encountered in power
  * ((kp * rhop * cpp)**(-0.5) + (ks * rhos * cps)**(-0.5))**(-1)

gas_phase.py:104: RuntimeWarning: overflow encountered in power
  cpgg[:, j] = Acp[j] + Bcp[j] * Tg + Ccp[j] * Tg**2 + Dcp[j] * Tg**3 + Ecp[j] * Tg**4

gas_phase.py:104: RuntimeWarning: invalid value encountered in add
  cpgg[:, j] = Acp[j] + Bcp[j] * Tg + Ccp[j] * Tg**2 + Dcp[j] * Tg**3 + Ecp[j] * Tg**4

/numpy/core/fromnumeric.py:87: RuntimeWarning: invalid value encountered in reduce
  return ufunc.reduce(obj, axis, dtype, out, **passkwargs)

gas_phase.py:172: RuntimeWarning: overflow encountered in double_scalars
  Ar = dp**3 * rhogi * (rhop - rhogi) * g / muin**2

gas_phase.py:175: RuntimeWarning: divide by zero encountered in double_scalars
  Umsr = (np.exp(-0.5405 * Lsi / Db) * (4.294e3 / Ar + 1.1) + 3.676e2 * Ar**(-1.5) + 1)

gas_phase.py:181: RuntimeWarning: invalid value encountered in double_scalars
  Rrb = (1 - 0.103 * (Umsr * umf - umf)**(-0.362) * Drbs)**(-1)

solid_phase.py:80: RuntimeWarning: overflow encountered in double_scalars
  Cs[i] = rhob_s[i] * cps[i]

solid_phase.py:161: RuntimeWarning: invalid value encountered in sqrt
  Nud = 2 + 0.6 * Re_dc**0.5 * Pr**0.33

kinetics.py:97: RuntimeWarning: invalid value encountered in power
  yv = m0v * Tsv**b0v

gas_phase.py:492: RuntimeWarning: invalid value encountered in sqrt
  Nud = 2 + 0.6 * Re_dc**0.5 * Pr**0.33

gas_phase.py:497: RuntimeWarning: invalid value encountered in power
  (7 - 10 * afg + 5 * afg**2) * (1 + 0.7 * Rep**0.2 * Pr**0.33)

gas_phase.py:498: RuntimeWarning: invalid value encountered in power
  + (1.33 - 2.4 * afg + 1.2 * afg**2) * Rep**0.7 * Pr**0.33

gas_phase.py:513: RuntimeWarning: invalid value encountered in power
  Nuf = 0.023 * ReD**0.8 * Pr**0.4

solid_phase.py:279: RuntimeWarning: invalid value encountered in power
  Nup = (7 - 10 * afg + 5 * afg**2) * (1 + 0.7 * Rep**0.2 * Pr**0.33) + (1.33 - 2.4 * afg + 1.2 * afg**2) * Rep**0.7 * Pr**0.33

solid_phase.py:362: RuntimeWarning: invalid value encountered in power
  Nup = (7 - 10 * afg + 5 * afg**2) * (1 + 0.7 * Rep**0.2 * Pr**0.33) + (1.33 - 2.4 * afg + 1.2 * afg**2) * Rep**0.7 * Pr**0.33

solid_phase.py:368: RuntimeWarning: overflow encountered in power
  qwr = np.pi * Dwi * epb / ((1 - ep) / (ep * epb) + (1 - ew) / ew + 1) * sc * (Tw**4 - Tp**4)

solid_phase.py:368: RuntimeWarning: invalid value encountered in subtract
  qwr = np.pi * Dwi * epb / ((1 - ep) / (ep * epb) + (1 - ew) / ew + 1) * sc * (Tw**4 - Tp**4)

solid_phase.py:381: RuntimeWarning: invalid value encountered in power
  Nuf = 0.023 * ReD**0.8 * Pr**0.4

solid_phase.py:433: RuntimeWarning: invalid value encountered in power
  24 / Re_dc * (1 + 8.1716 * Re_dc**(0.0964 + 0.5565 * sfc) * np.exp(-4.0655 * sfc))

solid_phase.py:464: RuntimeWarning: overflow encountered in multiply
  + Spav[0:Ni] * v[0:Ni] / rhos[0:Ni]

solid_phase.py:464: RuntimeWarning: invalid value encountered in add
  + Spav[0:Ni] * v[0:Ni] / rhos[0:Ni]

gas_phase.py:285: RuntimeWarning: invalid value encountered in power
  24 / Re_dc * (1 + 8.1716 * Re_dc**(0.0964 + 0.5565 * sfc) * np.exp(-4.0655 * sfc))

gas_phase.py:305: RuntimeWarning: overflow encountered in multiply
  Smgs = (3 / 4) * rhosbav * (rhog / rhos) * (Cd / ds) * np.abs(-ug - v)

kinetics.py:97: RuntimeWarning: overflow encountered in power
  yv = m0v * Tsv**b0v

kinetics.py:99: RuntimeWarning: invalid value encountered in true_divide
  xv = yv * Mv / np.sum(yv * Mv)
```

## Code

The system of ordinary differential equations (ODEs) is solved using the `solve_ivp()` function in SciPy. This solver function is called in `solve.py`. See `dydt_system.py` where the `dydt()` defines the system of ODEs.

## Reference

This code is based on the one-dimensional fluidized bed gasification model discussed in the Agu et al. 2019 paper.

Cornelius E. Agu, Christoph Pfeifer, Marianne Eikeland, Lars-Andre Tokheim, and Britt M.E. Moldestad. [**Detailed One-Dimensional Model for Steam-Biomass Gasification in a Bubbling Fluidized Bed**](https://pubs.acs.org/doi/10.1021/acs.energyfuels.9b01340). Energy & Fuels, no. 8, vol. 33, pp. 7385-7397, 2019.
