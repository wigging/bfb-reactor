import matplotlib.pyplot as plt
import numpy as np


def _style_axis(ax):
    ax.grid(color='0.9')
    ax.set_frame_on(False)
    ax.tick_params(color='0.9')


def plot_temp(Tg, Ts, x, t):
    Tsx = np.concatenate(([Ts[0, -1]], Ts[:, -1], [Ts[-1, -1]]))
    Tgx = np.concatenate(([Tg[0, -1]], Tg[:, -1], [Tg[-1, -1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x[0:75], Tsx[0:75] - 273)
    ax.plot(x, Tgx - 273)
    ax.set_xlabel('Height [m]')
    ax.set_ylabel('Temperature [°C]')
    _style_axis(ax)

    X, Y = np.meshgrid(t, x[1:-1], )
    _, ax = plt.subplots(subplot_kw={"projection": "3d"}, tight_layout=True)
    ax.plot_surface(X, Y, Tg - 273, cmap='viridis')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Height [m]')
    ax.set_zlabel('Temperature [K]')


def plot_bio_char(rhob_b, rhob_c, x):
    rhob_bx = np.concatenate(([rhob_b[0, -1]], rhob_b[:, -1], [rhob_b[-1, -1]]))
    rhob_cx = np.concatenate(([rhob_c[0, -1]], rhob_c[:, -1], [rhob_c[-1, -1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x[0:75], rhob_bx[0:75])
    ax.plot(x[0:75], rhob_cx[0:75])
    ax.set_xlabel('Height [m]')
    ax.set_ylabel('Concentration [kg/m³]')
    _style_axis(ax)


def plot_mfg(mfg, x):
    mfgx = np.concatenate(([mfg[0, -1]], mfg[:, -1], [mfg[-1, -1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x, mfgx)
    ax.set_xlabel('Height [m]')
    ax.set_ylabel('Gas flux [kg/(s⋅m²)]')
    _style_axis(ax)


def plot_h2(rhob_h2, t):
    _, ax = plt.subplots(tight_layout=True)
    ax.plot(t, rhob_h2[0], label='bottom')
    ax.plot(t, rhob_h2[-1], label='top')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('H₂ concentration [kg/m³]')
    ax.legend()
    _style_axis(ax)


def plot_velocity(mfg, rhob_g, v, x):
    ug = mfg / rhob_g
    ux = np.concatenate(([ug[0, -1]], ug[:, -1], [ug[-1, -1]]))
    vx = np.concatenate(([v[0, -1]], v[:, -1], [v[-1, -1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x, ux, label='gas')
    ax.plot(x, vx, label='solid')
    ax.set_xlabel('Height [m]')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend(loc='best')
    _style_axis(ax)


def show_plots():
    plt.show()
