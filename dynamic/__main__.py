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
        description='🚀 Bubbling fluidized bed (BFB) reactor program.',
        epilog='🤓 Enjoy.')

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

    print('\n🚀 Bubbling fluidized bed (BFB) reactor program.\n')

    if args.run:

        print('Run solver...\n')
        tic = time.perf_counter()

        # Get parameters and run the solver
        params = get_params(args.params)
        results = solver(params)

        # Elapsed time to get parameters and run solver
        toc = time.perf_counter()
        tsec = toc - tic
        print(f'\nelapsed time {tsec // 60:.0f}m {tsec % 60:.0f}s')

        # Create `results` folder to store results
        results_dir.mkdir(exist_ok=True)

        # Save results to binary NumPy files
        for key in results:
            np.save(f'{results_dir / key}', results[key])

        print(f'\nResults saved to {params_dir} folder.\n')

    if args.plot:

        print('Show plots...\n')

        # Get parameters
        params = get_params(args.params)

        # Load parameters from saved binary NumPy files
        x = np.load(results_dir / 'x.npy')
        t = np.load(results_dir / 't.npy')
        Ts = np.load(results_dir / 'Ts.npy')
        Tg = np.load(results_dir / 'Tg.npy')
        rhob_b = np.load(results_dir / 'rhob_b.npy')
        v = np.load(results_dir / 'v.npy')
        mfg = np.load(results_dir / 'mfg.npy')
        rhob_g = np.load(results_dir / 'rhob_g.npy')
        rhob_h2o = np.load(results_dir / 'rhob_h2o.npy')
        Tp = np.load(results_dir / 'Tp.npy')
        rhob_c = np.load(results_dir / 'rhob_c.npy')
        rhob_h2 = np.load(results_dir / 'rhob_h2.npy')
        rhob_ch4 = np.load(results_dir / 'rhob_ch4.npy')
        rhob_co = np.load(results_dir / 'rhob_co.npy')
        rhob_co2 = np.load(results_dir / 'rhob_co2.npy')
        rhob_t = np.load(results_dir / 'rhob_t.npy')
        rhob_ca = np.load(results_dir / 'rhob_ca.npy')
        Tw = np.load(results_dir / 'Tw.npy')

        # Plot results
        plotter.plot_temp(params, Tg, Tp, Ts, Tw, t, x)
        plotter.plot_bio_char(params, rhob_b, rhob_c, x)
        plotter.plot_mfg(mfg, x)
        plotter.plot_concentration(params, rhob_ch4, t, x, species='CH₄')
        plotter.plot_concentration(params, rhob_co, t, x, species='CO')
        plotter.plot_concentration(params, rhob_co2, t, x, species='CO₂')
        plotter.plot_concentration(params, rhob_h2, t, x, species='H₂')
        plotter.plot_concentration(params, rhob_h2o, t, x, species='H₂O')
        plotter.plot_concentration(params, rhob_t, t, x, species='Tar')
        plotter.plot_concentration(params, rhob_ca, t, x, species='Acc. char')
        plotter.plot_velocity(params, mfg, rhob_g, v, x)
        plotter.plot_surface(rhob_t, t, x, zlabel='Tar [kg/m3]')
        plotter.plot_surface(Tg - 273, t, x, zlabel='Tg [°C]')
        plotter.show_plots()

        print('Done.')


if __name__ == '__main__':
    main()
