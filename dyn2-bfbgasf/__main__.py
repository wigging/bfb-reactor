import plotter
from parameters import get_params
from solver import solver


def main():
    """
    Run the 1D bubbling fluidized bed (BFB) gasification model.
    """
    params = get_params('dyn2-bfbgasf/params.json')
    results = solver(params)

    plotter.plot_ts(results)
    plotter.plot_tg(results)
    plotter.plot_mfg(results)
    plotter.show_plots()


if __name__ == '__main__':
    main()
