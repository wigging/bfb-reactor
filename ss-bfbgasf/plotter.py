import matplotlib.pyplot as plt
import numpy as np


def _style_axis(ax):
    ax.grid(color='0.9')
    ax.set_frame_on(False)
    ax.tick_params(color='0.9')


def plot_ug_v(results):
    ug = results['ug']
    ugin = results['ugin']
    v = results['v']
    z = results['z']

    ugz = np.concatenate(([ugin], ug))
    vz = np.concatenate((v, [v[-1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(z, ugz, label='ug')
    ax.plot(z, vz, label='v')
    ax.legend(loc='best')
    ax.set_xlabel('Bed height, z [m]')
    ax.set_ylabel('Velocity [m/s]')
    _style_axis(ax)


def plot_mfg(results):
    mfg = results['mfg']
    mfgin = results['mfgin']
    z = results['z']

    mfgz = np.concatenate(([mfgin], mfg))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(z, mfgz)
    ax.set_xlabel('Bed height, z [m]')
    ax.set_ylabel('Gas mass flux, mfg [kg/m²⋅s]')
    _style_axis(ax)


def plot_rhoba_rhobb_rhobc(results):
    """
    Plot biomass and char concentrations.
    """
    rhoba = results['rhoba']
    rhobb = results['rhobb']
    rhobbin = results['rhobbin']
    rhobc = results['rhobc']
    z = results['z']

    rhobaz = np.concatenate((rhoba, [0]))
    rhobbz = np.concatenate((rhobb, [rhobbin]))
    rhobcz = np.concatenate((rhobc, [0]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(z, rhobaz, label=r'ρ̅𝖺')
    ax.plot(z, rhobbz, label=r'ρ̅𝖻')
    ax.plot(z, rhobcz, label='ρ̅𝖼')
    ax.set_xlabel('Bed height, z [m]')
    ax.set_ylabel('Concentration, ρ̅ [kg/m³]')
    ax.legend(loc='best')
    _style_axis(ax)


def show_plots():
    plt.show()
