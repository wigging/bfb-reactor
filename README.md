# One-dimensional BFB gasifier model

:warning: This project is a work in progress. :warning:

This repository contains a dynamic and a steady-state 1D model of a bubbling fluidized bed (BFB) gasifier. At the moment, the dynamic model is in the `dyn-bfbgasf` folder while the steady-state model is in the `ss-bfbgasf` folder. Another attempt of the dynamic model is `dyn2-bfbgasf`.

## Usage

Use the commands shown below to run each model.

```bash
# Run the dynamic model
$ cd dyn-bfbgasf
$ python main.py

# Run the other dynamic model
$ python dyn2-bfbgasf
```

```bash
# Run the steady-state model
$ python ss-bfbgasf
```

## Reference

This code is based on the one-dimensional fluidized bed gasification model discussed in the Agu et al. 2019 paper.

Cornelius E. Agu, Christoph Pfeifer, Marianne Eikeland, Lars-Andre Tokheim, and Britt M.E. Moldestad. [**Detailed One-Dimensional Model for Steam-Biomass Gasification in a Bubbling Fluidized Bed**](https://pubs.acs.org/doi/10.1021/acs.energyfuels.9b01340). Energy & Fuels, no. 8, vol. 33, pp. 7385-7397, 2019.
