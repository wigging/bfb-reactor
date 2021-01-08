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


def show_plots():
    plt.show()
