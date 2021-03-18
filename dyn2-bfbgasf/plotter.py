import matplotlib.pyplot as plt
import numpy as np


def _style_axis(ax):
    ax.grid(color='0.9')
    ax.set_frame_on(False)
    ax.tick_params(color='0.9')


def plot_mfg(results):
    x = results['x']
    t = results['t']
    mfg = results['mfg']

    # mfg along reactor at final time
    mfgx = np.concatenate(([0.2], mfg[:, -1], [mfg[-1, -1]]))

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
