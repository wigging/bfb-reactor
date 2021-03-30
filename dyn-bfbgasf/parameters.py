import json
import numpy as np


def get_params(json_file):
    """
    Read JSON file containing the model parameters. Commented lines that begin
    with // are ignored. JSON data is returned as a dictionary. Calculate
    inlet parameters and add them to the dictionary.
    """
    R = 8.314
    g = 9.81

    # Read JSON file and store parameters in a dictionary
    json_str = ''

    with open(json_file) as jfile:
        for line in jfile:
            if '//' not in line:
                json_str += line

    json_dict = json.loads(json_str)

    # Get parameters from JSON dictionary
    Db = json_dict['Db']
    N1 = json_dict['N1']
    N2 = json_dict['N2']
    N3 = json_dict['N3']
    Ls = json_dict['Ls']
    Mgin = json_dict['Mgin']
    Pa = json_dict['Pa']
    SB = json_dict['SB']
    Tgin = json_dict['Tgin']

    ef0 = json_dict['ef0']
    emf = json_dict['emf']
    rhob = json_dict['rhob']
    rhoc = json_dict['rhoc']
    rhop = json_dict['rhop']
    wa = json_dict['wa']
    wc = json_dict['wc']

    # Biomass shrinkage factor [-]
    psi = rhoc / (rhob * (wc + wa))

    # Total grid points (N) and grid points to bed top (Np)
    N = N1 + N2 + N3
    Np = N1 + N2

    # Bed cross-sectional area [m²]
    Ab = (np.pi / 4) * (Db**2)

    # Inlet gas mass flux [kg/(s⋅m²)]
    msdot = json_dict['msdot'] / 3600
    mfgin = SB * msdot / Ab

    # Inlet gas velocity [m/s]
    rhogin = json_dict['P'] * Mgin / (R * Tgin) * 1e-3
    ugin = mfgin / rhogin

    # Bulk gas density at inlet [kg/m³]
    Pin = (1 - ef0) * rhop * g * Ls + Pa
    rhog_in = Pin * Mgin / (R * Tgin) * 1e-3
    rhob_gin = rhog_in

    # here
    Lmf = (1 - ef0) / (1 - emf) * Ls

    # here
    Pin = (1 - ef0) * rhop * g * Ls + Pa

    # Reactor internal diameter, same as Db [m]
    Dwo = json_dict['Dwo']
    xw = json_dict['xw']
    Dwi = Dwo - 2 * xw

    # add calculated parameters to JSON dictionary
    json_dict['Ab'] = Ab
    json_dict['Dwi'] = Dwi
    json_dict['Lmf'] = Lmf
    json_dict['N'] = N
    json_dict['Np'] = Np
    json_dict['Pin'] = Pin
    json_dict['mfgin'] = mfgin
    json_dict['psi'] = psi
    json_dict['rhob_gin'] = rhob_gin
    json_dict['ugin'] = ugin

    # return all parameters along with calculated parameters
    return json_dict
