import numpy as n
import dpt_tools as dpt

import matplotlib.pyplot as plt

E = n.linspace(0.0, 2.0*n.pi, num=100, dtype=n.float)
e = n.linspace(0, 0.99, num=100, dtype=n.float)

Ev, ev = n.meshgrid(E, e)

Mv = dpt.eccentric2mean(Ev, ev)

E0 = dpt.kepler_guess(Mv, ev)


fig, ax = plt.subplots()
CS = ax.contour(Ev*180.0/n.pi, ev, n.abs(Ev - E0)*180.0/n.pi)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel('E [deg]')
ax.set_ylabel('e')

E = n.linspace(1.7, 2.7, num=500, dtype=n.float)
e = n.linspace(0.8, 0.99, num=500, dtype=n.float)

Ev, ev = n.meshgrid(E, e)

Mv = dpt.eccentric2mean(Ev, ev)

E0 = dpt.kepler_guess(Mv, ev)


fig, ax = plt.subplots()
CS = ax.contour(Ev*180.0/n.pi, ev, n.abs(Ev - E0)*180.0/n.pi)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel('E [deg]')
ax.set_ylabel('e')

E = n.linspace(2.45, 2.65, num=500, dtype=n.float)
e = n.linspace(0.95, 0.9999, num=500, dtype=n.float)

Ev, ev = n.meshgrid(E, e)

Mv = dpt.eccentric2mean(Ev, ev)

E0 = dpt.kepler_guess(Mv, ev)


fig, ax = plt.subplots()
CS = ax.contour(Ev*180.0/n.pi, ev, n.abs(Ev - E0)*180.0/n.pi)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel('E [deg]')
ax.set_ylabel('e')


plt.show()