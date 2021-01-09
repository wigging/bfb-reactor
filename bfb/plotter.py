import matplotlib.pyplot as plt


def plot_v(results):
    """
    Plot solid velocity over time.
    """
    t = results['t']
    v = results['v']

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(t, v[0], label='v[0]')
    ax.plot(t, v[-1], label='v[-1]')
    ax.set_xlabel('time, t [s]')
    ax.set_ylabel('velocity, v [m/s]')
    ax.legend()


def plot_rhobg(results):
    """
    Plot bulk gas mass concentration over time.
    """
    t = results['t']
    rhobg = results['rhob_g']

    _, ax = plt.subplots(tight_layout=True)
    ax.plot(t, rhobg[0], label='rhobg[0]')
    ax.plot(t, rhobg[-1], label='rhobg[-1]')
    ax.set_xlabel('time, t [s]')
    ax.set_ylabel('bulk gas mass concentration, ρ̅𝗀 [kg/m³]')
    ax.legend()


def show_plots():
    plt.show()
