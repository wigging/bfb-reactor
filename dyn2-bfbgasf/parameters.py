import json
import numpy as np


def get_params(json_file):
    """
    Read JSON file containing the model parameters. Commented lines that begin
    with // are ignored. JSON data is returned as a dictionary. Calculate
    inlet parameters and add them to the dictionary.
    """
    json_str = ''

    with open(json_file) as jfile:
        for line in jfile:
            if '//' not in line:
                json_str += line

    json_dict = json.loads(json_str)

    # total grid points (N) and grid points to bed top (Np)
    N = json_dict['N1'] + json_dict['N2'] + json_dict['N3']
    Np = json_dict['N1'] + json_dict['N2']

    # bed cross-sectional area [m²]
    Db = json_dict['Db']
    Ab = (np.pi / 4) * (Db**2)

    # inlet gas mass flux [kg/(s⋅m²)]
    SB = json_dict['SB']
    msdot = json_dict['msdot'] / 3600
    mfgin = SB * msdot / Ab

    # bulk gas density at inlet [kg/m³]
    Ls = json_dict['Ls']
    Mgin = json_dict['Mgin']
    Pa = json_dict['Pa']
    Tgin = json_dict['Tgin']
    ef0 = json_dict['ef0']
    rhop = json_dict['rhop']
    R = 8.314
    g = 9.81
    Pin = (1 - ef0) * rhop * g * Ls + Pa
    rhog_in = Pin * Mgin / (R * Tgin) * 1e-3
    rhob_gin = rhog_in

    # reactor internal diameter, same as Db [m]
    Dwo = json_dict['Dwo']
    xw = json_dict['xw']
    Dwi = Dwo - 2 * xw

    # add calculated parameters to JSON dictionary
    json_dict['Dwi'] = Dwi
    json_dict['N'] = N
    json_dict['Np'] = Np
    json_dict['mfgin'] = mfgin
    json_dict['rhob_gin'] = rhob_gin

    # return all parameters along with calculated parameters
    return json_dict
