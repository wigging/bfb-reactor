# One-dimensional BFB reactor model

:warning: This project is a work in progress. :warning:

This repository contains a dynamic and a steady-state 1D model of a bubbling fluidized bed (BFB) reactor. At the moment, the dynamic model is in the `dynamic` folder while the steady-state model is in the `steady` folder.

## Usage

Use the commands shown below to run each model.

```bash
# Run the dynamic model
$ python dynamic examples/params.json --run

# Plot results from the dynamic model
$  python dynamic examples/params.json --plot
```

```bash
# Run the steady-state model
$ python steady
```

## Reference

This code is based on the one-dimensional fluidized bed gasification model discussed in the Agu et al. 2019 paper.

Cornelius E. Agu, Christoph Pfeifer, Marianne Eikeland, Lars-Andre Tokheim, and Britt M.E. Moldestad. [**Detailed One-Dimensional Model for Steam-Biomass Gasification in a Bubbling Fluidized Bed**](https://pubs.acs.org/doi/10.1021/acs.energyfuels.9b01340). Energy & Fuels, no. 8, vol. 33, pp. 7385-7397, 2019.
