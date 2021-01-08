from solver import solver
from parameters import Parameters
from plotter import plot_v, show_plots


def main():
    """
    Run a one-dimensional bubbling fluidized bed (BFB) biomass gasification
    model and plot the results.
    """

    # Get parameters from JSON file
    params = Parameters('params.json')

    # Run solver and get results
    results = solver(params)

    # Plot results
    plot_v(results)
    show_plots()


if __name__ == '__main__':
    main()
