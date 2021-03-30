import matplotlib.pyplot as plt
import numpy as np


def _style_axis(ax):
    ax.grid(color='0.9')
    ax.set_frame_on(False)
    ax.tick_params(color='0.9')


def plot_temp(params, Tg, Tp, Ts, Tw, t, x):
    """
    Plot temperatures along reactor height at final time. Surface plot of gas
    temperature with respect to time and reactor height.
    """
    Np = params['Np']

    Twx = np.concatenate(([Tw[0, -1]], Tw[:, -1], [Tw[-1, -1]]))
    Tsx = np.concatenate(([Ts[0, -1]], Ts[:, -1], [Ts[-1, -1]]))
    Tpx = np.concatenate(([Tp[0, -1]], Tp[:, -1], [Tp[-1, -1]]))
    Tgx = np.concatenate(([Tg[0, -1]], Tg[:, -1], [Tg[-1, -1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x[0:Np], Tsx[0:Np] - 273, 'k--', label='Ts')
    ax.plot(x, Tgx - 273, 'k', label='Tg')
    ax.set_xlabel('Reactor height [m]')
    ax.set_ylabel('Temperature [°C]')
    ax.legend(loc='best')
    _style_axis(ax)

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x[0:Np], Twx[0:Np] - 273, 'r.', label='Tw')
    ax.plot(x[0:Np], Tsx[0:Np] - 273, 'k--', label='Ts')
    ax.plot(x[0:Np], Tpx[0:Np] - 273, 'bo', fillstyle='none', label='Tp')
    ax.plot(x[0:Np], Tgx[0:Np] - 273, 'k', label='Tg')
    ax.set_xlabel('Reactor height [m]')
    ax.set_ylabel('Bed temperature [°C]')
    ax.legend(loc='best')
    _style_axis(ax)

    X, Y = np.meshgrid(t, x[1:-1], )
    _, ax = plt.subplots(subplot_kw={"projection": "3d"}, tight_layout=True)
    ax.plot_surface(X, Y, Tg - 273, cmap='viridis')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Reactor height [m]')
    ax.set_zlabel('Tg [°C]')


def plot_bio_char(params, rhob_b, rhob_c, x):
    """
    Plot biomass and char concentrations along reactor height at final time.
    """
    Np = params['Np']

    rhob_bx = np.concatenate(([rhob_b[0, -1]], rhob_b[:, -1], [rhob_b[-1, -1]]))
    rhob_cx = np.concatenate(([rhob_c[0, -1]], rhob_c[:, -1], [rhob_c[-1, -1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x[0:Np], rhob_bx[0:Np], label='biomass')
    ax.plot(x[0:Np], rhob_cx[0:Np], label='char')
    ax.set_xlabel('Reactor height [m]')
    ax.set_ylabel('Concentration [kg/m³]')
    ax.legend(loc='best')
    _style_axis(ax)


def plot_mfg(mfg, x):
    """
    Plot gas mass flux along reactor height at final time.
    """
    mfgx = np.concatenate(([mfg[0, -1]], mfg[:, -1], [mfg[-1, -1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x, mfgx)
    ax.set_xlabel('Reactor height [m]')
    ax.set_ylabel('Gas mass flux [kg/(s⋅m²)]')
    _style_axis(ax)


def plot_h2(params, rhob_h2, t):
    """
    Plot H₂ concentration with respect to time.
    """
    Np = params['Np']

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(t, rhob_h2[0], 'k', label='bottom')
    ax.plot(t, rhob_h2[Np], 'k-.', label='bed top')
    ax.plot(t, rhob_h2[-1], 'k--', label='top')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('H₂ concentration [kg/m³]')
    ax.legend()
    _style_axis(ax)


def plot_velocity(params, mfg, rhob_g, v, x):
    """
    Plot gas and solid velocities along reactor height at final time.
    """
    Np = params['Np']

    ug = mfg / rhob_g
    ux = np.concatenate(([ug[0, -1]], ug[:, -1], [ug[-1, -1]]))
    vx = np.concatenate(([v[0, -1]], v[:, -1], [v[-1, -1]]))

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(x[0:Np], vx[0:Np], 'k--', label='v')
    ax.plot(x, ux, 'k', label='ug')
    ax.set_xlabel('Reactor height [m]')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend(loc='best')
    _style_axis(ax)


def show_plots():
    plt.show()
