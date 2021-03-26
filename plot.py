import matplotlib.pyplot as plt
import numpy as np

x = np.load('results/x.npy')
Ts = np.load('results/Ts.npy')
Tg = np.load('results/Tg.npy')

Tsx = np.concatenate(([Ts[0, -1]], Ts[:, -1], [Ts[-1, -1]]))
Tgx = np.concatenate(([Tg[0, -1]], Tg[:, -1], [Tg[-1, -1]]))

_, ax = plt.subplots()
ax.plot(x[0:75], Tsx[0:75] - 273)
ax.plot(x, Tgx - 273)
ax.set_xlabel('Height [m]')
ax.set_ylabel('Temperature [°C]')
ax.grid(color='0.9')
ax.set_frame_on(False)
ax.tick_params(color='0.9')

plt.show()
