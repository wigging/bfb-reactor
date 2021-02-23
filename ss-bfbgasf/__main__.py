import json

import plotter
from solver import solver


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

    json_dict = json.loads(json_str)
    return json_dict


def main():
    """
    Run a 1-D bubbling fluidized bed (BFB) steady-state gasification model.
    """

    params = _get_params('ss-bfbgasf/params.json')
    results = solver(params)

    plotter.plot_ug_v(results)
    plotter.plot_mfg(results)
    plotter.plot_rhoab_rhobb_rhocb(results)
    plotter.show_plots()


if __name__ == '__main__':
    main()
