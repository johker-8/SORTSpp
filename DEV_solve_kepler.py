import numpy as n
import dpt_tools as dpt
import matplotlib.pyplot as plt

E = n.linspace(0.0, 2.0*n.pi, num=300, dtype=n.float)
e = n.linspace(0, 0.99, num=500, dtype=n.float)

Ev, ev = n.meshgrid(E, e)
Mv = dpt.eccentric2mean(Ev, ev)

it_num = n.empty(Mv.shape, dtype=n.int)
err = n.empty(Mv.shape, dtype=n.float)
f_eval = n.empty(Mv.shape, dtype=n.float)

for I, eit in enumerate(e):
    for J, Eit in enumerate(E):
        M = dpt.eccentric2mean(Eit, eit)

        E0 = dpt.kepler_guess(M, eit)
        E_calc, it = dpt.laguerre_solve_kepler(E0, M, eit, tol=1e-12)


        f_eval[I,J] = n.abs(M - E_calc + eit*n.sin(E_calc))
        it_num[I,J] = it
        err[I,J] = n.abs(Eit - E_calc)

fig, ax = plt.subplots()
CS = ax.contour(Ev*180.0/n.pi, ev, err*180.0/n.pi)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel('E [deg]')
ax.set_ylabel('e')

fig, ax = plt.subplots()
CS = ax.contour(Mv*180.0/n.pi, ev, err*180.0/n.pi)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel('M [deg]')
ax.set_ylabel('e')

fig, ax = plt.subplots()
CS = ax.contour(Mv*180.0/n.pi, ev, f_eval*180.0/n.pi)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel('M [deg]')
ax.set_ylabel('e')

fig, ax = plt.subplots()
CS = ax.contour(Ev*180.0/n.pi, ev, it_num, levels=n.arange(0,n.max(it_num.flatten())))
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel('E [deg]')
ax.set_ylabel('e')

plt.show()