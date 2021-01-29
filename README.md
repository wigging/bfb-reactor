# One-dimensional BFB gasifier model

:warning: This project is a work in progress.

## Usage

Execute the `main.py` file to run the model.

```bash
$ python main.py
```

## Code

The system of ordinary differential equations (ODEs) is solved using the `solve_ivp()` function in SciPy. This solver function is called in `solve.py`. See `dydt_system.py` where the `dydt()` defines the system of ODEs.

## Reference

This code is based on the one-dimensional fluidized bed gasification model discussed in the Agu et al. 2019 paper.

Cornelius E. Agu, Christoph Pfeifer, Marianne Eikeland, Lars-Andre Tokheim, and Britt M.E. Moldestad. [**Detailed One-Dimensional Model for Steam-Biomass Gasification in a Bubbling Fluidized Bed**](https://pubs.acs.org/doi/10.1021/acs.energyfuels.9b01340). Energy & Fuels, no. 8, vol. 33, pp. 7385-7397, 2019.
