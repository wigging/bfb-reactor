import matplotlib.pyplot as plt
import numpy as np


def _style_axis(ax):
    ax.grid(color='0.9')
    ax.set_frame_on(False)
    ax.tick_params(color='0.9')


def plot_ts(results):
    x = results['x']
    t = results['t']
    Ts = results['Ts']

    # Ts along height of the reactor at final time
    Tsx = np.concatenate(([Ts[0, -1]], Ts[:, -1], [Ts[-1, -1]]))

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, tight_layout=True)

    ax1.plot(t, Ts[0], label='btm')
    ax1.plot(t, Ts[50], label='mid')
    ax1.plot(t, Ts[-1], label='top')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('Ts [K]')
    ax1.legend()
    _style_axis(ax1)

    ax2.plot(x, Tsx)
    ax2.set_xlabel('x [m]')
    _style_axis(ax2)


def plot_tg(results):
    x = results['x']
    t = results['t']
    Tg = results['Tg']

    # Tg along height of the reactor at final time
    Tgx = np.concatenate(([Tg[0, -1]], Tg[:, -1], [Tg[-1, -1]]))

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, tight_layout=True)

    ax1.plot(t, Tg[0], label='btm')
    ax1.plot(t, Tg[50], label='mid')
    ax1.plot(t, Tg[-1], label='top')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('Tg [K]')
    ax1.legend()
    _style_axis(ax1)

    ax2.plot(x, Tgx)
    ax2.set_xlabel('x [m]')
    _style_axis(ax2)


def plot_rhobb(results):
    x = results['x']
    t = results['t']
    rhob_b = results['rhob_b']

    # rhob_b along height of the reactor at final time
    rhob_bx = np.concatenate(([rhob_b[0, -1]], rhob_b[:, -1], [rhob_b[-1, -1]]))

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, tight_layout=True)

    ax1.plot(t, rhob_b[0], label='btm')
    ax1.plot(t, rhob_b[50], label='mid')
    ax1.plot(t, rhob_b[-1], label='top')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('rhob_b [kg/m³]')
    ax1.legend()
    _style_axis(ax1)

    ax2.plot(x, rhob_bx)
    ax2.set_xlabel('x [m]')
    _style_axis(ax2)


def plot_mfg(results):
    x = results['x']
    t = results['t']
    mfg = results['mfg']

    # mfg along reactor at final time
    mfgx = np.concatenate(([mfg[0, -1]], mfg[:, -1], [mfg[-1, -1]]))

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, tight_layout=True)

    ax1.plot(t, mfg[0], label='btm')
    ax1.plot(t, mfg[50], label='mid')
    ax1.plot(t, mfg[-1], label='top')
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('mfg [kg/(s⋅m²)]')
    ax1.legend()
    _style_axis(ax1)

    ax2.plot(x, mfgx)
    ax2.set_xlabel('x [m]')
    _style_axis(ax2)


def show_plots():
    plt.show()
