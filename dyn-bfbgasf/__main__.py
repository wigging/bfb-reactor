import argparse
import pathlib
import numpy as np
import time

import plotter
from parameters import get_params
from solver import solver


def _command_line_args():
    """
    Command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='🚀 Bubbling fluidized bed (BFB) gasifier program.',
        epilog='🤓 Enjoy the program.')

    parser.add_argument('params', help='path to JSON parameters file')
    parser.add_argument('-r', '--run', action='store_true', help='run the program')
    parser.add_argument('-p', '--plot', action='store_true', help='plot results')

    args = parser.parse_args()
    return args


def main():
    """
    Run the 1D bubbling fluidized bed (BFB) gasification model.
    """
    args = _command_line_args()
    params_dir = pathlib.Path(args.params).parent
    results_dir = pathlib.Path(params_dir / 'results')

    if args.run:

        tic = time.perf_counter()

        # Get parameters and solve model
        params = get_params(args.params)
        results = solver(params)

        # Elapsed time to get parameters and solve model
        toc = time.perf_counter()
        tsec = toc - tic
        print(f'\n{"elapsed time:":10} {tsec // 60:.0f}m {tsec % 60:.0f}s')

        # Create `results` folder to store results
        results_dir.mkdir(exist_ok=True)

        # Save results to binary NumPy files
        for key in results:
            np.save(f'{results_dir / key}', results[key])

    if args.plot:

        # Load parameters from saved binary NumPy files
        x = np.load(results_dir / 'x.npy')
        t = np.load(results_dir / 't.npy')
        Ts = np.load(results_dir / 'Ts.npy')
        Tg = np.load(results_dir / 'Tg.npy')
        rhob_b = np.load(results_dir / 'rhob_b.npy')
        mfg = np.load(results_dir / 'mfg.npy')
        rhob_c = np.load(results_dir / 'rhob_c.npy')
        rhob_h2 = np.load(results_dir / 'rhob_h2.npy')

        # Plot results
        plotter.plot_temp(Tg, Ts, x, t)
        plotter.plot_bio_char(rhob_b, rhob_c, x)
        plotter.plot_mfg(mfg, x)
        plotter.plot_h2(rhob_h2, t)
        plotter.show_plots()

    # np.save('results/x', results['x'])
    # np.save('results/t', results['t'])
    # np.save('results/Ts', results['Ts'])
    # np.save('results/Tg', results['Tg'])
    # np.save('results/rhob_b', results['rhob_b'])
    # np.save('results/v', results['v'])
    # np.save('results/mfg', results['mfg'])
    # np.save('results/rhob_g', results['rhob_g'])
    # np.save('results/rhob_h2o', results['rhob_h2o'])
    # np.save('results/Tp', results['Tp'])
    # np.save('results/rhob_c', results['rhob_c'])
    # np.save('results/rhob_h2', results['rhob_h2'])
    # np.save('results/rhob_ch4', results['rhob_ch4'])
    # np.save('results/rhob_co', results['rhob_co'])
    # np.save('results/rhob_co2', results['rhob_co2'])
    # np.save('results/rhob_t', results['rhob_t'])
    # np.save('results/rhob_ca', results['rhob_ca'])
    # np.save('results/Tw', results['Tw'])


if __name__ == '__main__':
    main()
