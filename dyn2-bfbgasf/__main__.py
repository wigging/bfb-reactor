import pathlib
import numpy as np
import time

import plotter
from parameters import get_params
from solver import solver


def main():
    """
    Run the 1D bubbling fluidized bed (BFB) gasification model.
    """
    tic = time.perf_counter()

    # Get parameters and solve model
    params = get_params('dyn2-bfbgasf/params.json')
    results = solver(params)

    # Elapsed time to get parameters and solve model
    toc = time.perf_counter()
    tsec = toc - tic
    print(f'\n{" Summary ":-^60}\n')
    print(f'{"elapsed time:":10} {tsec // 60:.0f}m {tsec % 60:.0f}s')

    # Save results to binary NumPy files
    pathlib.Path('./results').mkdir(exist_ok=True)
    np.save('results/x', results['x'])
    np.save('results/t', results['t'])
    np.save('results/Ts', results['Ts'])
    np.save('results/Tg', results['Tg'])
    np.save('results/rhob_b', results['rhob_b'])
    np.save('results/v', results['v'])
    np.save('results/mfg', results['mfg'])
    np.save('results/rhob_g', results['rhob_g'])
    np.save('results/rhob_h2o', results['rhob_h2o'])
    np.save('results/Tp', results['Tp'])
    np.save('results/rhob_c', results['rhob_c'])
    np.save('results/rhob_h2', results['rhob_h2'])
    np.save('results/rhob_ch4', results['rhob_ch4'])
    np.save('results/rhob_co', results['rhob_co'])
    np.save('results/rhob_co2', results['rhob_co2'])
    np.save('results/rhob_t', results['rhob_t'])
    np.save('results/rhob_ca', results['rhob_ca'])
    np.save('results/Tw', results['Tw'])

    # Plot results
    # plotter.plot_ts(results)
    # plotter.plot_tg(results)
    # plotter.plot_rhobb(results)
    # plotter.plot_mfg(results)
    # plotter.show_plots()


if __name__ == '__main__':
    main()
