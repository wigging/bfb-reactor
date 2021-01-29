import json
import time

import plotter
from solver import solver
from parameters import Parameters


def _get_params(json_file):
    """
    Get parameters from JSON file. Commented lines in the JSON file that begin
    with // are ignored. Parameters are returned as a dictionary.
    """
    json_str = ''

    with open(json_file) as jfile:
        for line in jfile:
            if '//' not in line:
                json_str += line

    json_data = json.loads(json_str)
    return json_data


def main():
    """
    Run a one-dimensional bubbling fluidized bed (BFB) biomass gasification
    model and plot the results.
    """
    tic = time.perf_counter()

    # Get parameters from JSON file
    params_dict = _get_params('params.json')
    params = Parameters(params_dict)

    # Get results from the solver
    results = solver(params)

    # Print execution time for the program
    toc = time.perf_counter()
    tsec = toc - tic
    print(f'\nExecution time = {tsec // 60:.0f}m {tsec % 60:.0f}s')

    # Plot results
    plotter.plot_rhobb(results)
    plotter.plot_ts(results)
    plotter.plot_v(results)
    plotter.show_plots()


if __name__ == '__main__':
    main()
