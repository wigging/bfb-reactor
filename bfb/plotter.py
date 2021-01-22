import matplotlib.pyplot as plt


def plot_rhobb(results):
    t = results['t']
    rhob_b = results['rhob_b']

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(t, rhob_b[0], label='[0]')
    ax.plot(t, rhob_b[50], label='[50]')
    ax.plot(t, rhob_b[-1], label='[-1]')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('rhob_b [kg/m³]')
    ax.legend()


def plot_ts(results):
    t = results['t']
    Ts = results['Ts']

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(t, Ts[0], label='[0]')
    ax.plot(t, Ts[50], label='[50]')
    ax.plot(t, Ts[-1], label='[-1]')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Ts [K]')
    ax.legend()


def plot_v(results):
    t = results['t']
    v = results['v']

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(t, v[0], label='[0]')
    ax.plot(t, v[50], label='[50]')
    ax.plot(t, v[-1], label='[-1]')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('v [m/s]')
    ax.legend()


def show_plots():
    plt.show()
