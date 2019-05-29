#!/usr/bin/env python
#
#
import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from mpi4py import MPI
import sys
import os
import time

comm = MPI.COMM_WORLD

# SORTS imports CORE
import population_library as plib
from simulation import Simulation
from population import Population

#SORTS Libraries
import radar_library as rlib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib

import propagator_sgp4
import dpt_tools as dpt
import plothelp
import space_object

from catalogue import Catalogue

os.environ['TZ'] = 'GMT'
time.tzset()

####### RUN CONFIG #######
scenario = 0
##########################

event_date = np.datetime64('2018-03-27T05:40')
event_mjd = dpt.npdt2mjd(event_date)


so_list = [
  space_object.SpaceObject(
    a=6800.0,
    e=0.01,
    i=98.0,
    aop=56.0,
    raan=79.0,
    mu0=220.0,
    mjd0=event_mjd,
  ),
  space_object.SpaceObject(
    a=6800.0,
    e=0.01,
    i=98.0,
    aop=56.0,
    raan=199.0,
    mu0=250.0,
    mjd0=event_mjd,
  ),
]

tnum = 1000

t = np.linspace(0, 3600.0*1.0, num=tnum, dtype=np.float64)
el = np.linspace(-90, 90, num=len(t), dtype=np.float64)

states = []
for so in so_list:
    states.append(so.get_orbit(t))

plt.style.use('dark_background')

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plothelp.draw_earth_grid(ax, alpha = 0.2, color='white')

ax.view_init(elev=10.0, azim=0.0)

ax.grid(False)
plt.axis('off')

def gen_ellipsoid(center, radii, num=20):
    u = np.linspace(0.0, 2.0 * np.pi, num)
    v = np.linspace(0.0, np.pi, num)
    x = radii * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return x.flatten(), y.flatten(), z.flatten()
    

#traj
ax_traj_list = []
ax_point_list = []
ax_cov_list = []
for ind in range(len(so_list)):
    ax_traj, = ax.plot([],[],[], '-', alpha=0.5, color="white")
    ax_traj_list.append(ax_traj)

    ax_point, = ax.plot([],[],[], '.', alpha=1, color="blue")
    ax_point_list.append(ax_point)
    
    x, y, z = gen_ellipsoid([0,0,0], 1.0)
    ax_cov, = ax.plot(x, y, z, '.b', alpha=0.2)
    ax_cov_list.append(ax_cov)

so_ax = fig.add_axes([.65, .05, .33, .25], facecolor='k', projection='3d')
plothelp.draw_earth_grid(so_ax, alpha = 0.2, color='white')


so_ax_traj_list = []
so_ax_point_list = []
so_ax_cov_list = []
for ind in range(len(so_list)):
    ax_traj, = so_ax.plot([],[],[], '-', alpha=0.5, color="white")
    so_ax_traj_list.append(ax_traj)

    ax_point, = so_ax.plot([],[],[], '.', alpha=1, color="r")
    so_ax_point_list.append(ax_point)
    
    x, y, z = gen_ellipsoid([0,0,0], 1.0)
    ax_cov, = so_ax.plot(x, y, z, '.b', alpha=0.2)
    so_ax_cov_list.append(ax_cov)

so_ax.set(
    title='Thalia',
)
_lim = 5.0e3

cov_size = 3e3

delta = 7500e3
ax.set_xlim([-delta,delta])
ax.set_ylim([-delta,delta])
ax.set_zlim([-delta,delta])

def run(tt_ind):
    # update the data

    titl.set_text('Simulation: conjunction event %.4f h' % (t[tt_ind]/3600.0))

    print('Updating plot frame {} of {}'.format(tt_ind, len(t)))
    for ind in range(len(so_list)):

        ax_traj_list[ind].set_data(
            states[ind][0, :(tt_ind+1)],
            states[ind][1, :(tt_ind+1)],
        )
        ax_traj_list[ind].set_3d_properties(
            states[ind][2, :(tt_ind+1)],
        )
        ax_traj_list[ind].figure.canvas.draw()

        ax_point_list[ind].set_data(
            states[ind][0, tt_ind],
            states[ind][1, tt_ind],
        )
        ax_point_list[ind].set_3d_properties(
            states[ind][2, tt_ind],
        )
        ax_point_list[ind].figure.canvas.draw()
    
        x, y, z = gen_ellipsoid(states[ind][:, tt_ind], cov_size)
        
        ax_cov_list[ind].set_data(x, y)
        ax_cov_list[ind].set_3d_properties(z)
        ax_cov_list[ind].figure.canvas.draw()

        so_ax_cov_list[ind].set_data(x, y)
        so_ax_cov_list[ind].set_3d_properties(z)
        so_ax_cov_list[ind].figure.canvas.draw()

    
    so_ax.set_xlim([states[0][0, tt_ind]-_lim,states[0][0, tt_ind]+_lim])
    so_ax.set_ylim([states[0][1, tt_ind]-_lim,states[0][1, tt_ind]+_lim])
    so_ax.set_zlim([states[0][2, tt_ind]-_lim,states[0][2, tt_ind]+_lim])
    
    so_ax.view_init(elev=el[tt_ind], azim=0.0)
    
    for ind in range(len(so_list)):

        so_ax_traj_list[ind].set_data(
            states[ind][0, :(tt_ind+1)],
            states[ind][1, :(tt_ind+1)],
        )
        so_ax_traj_list[ind].set_3d_properties(
            states[ind][2, :(tt_ind+1)],
        )
        so_ax_traj_list[ind].figure.canvas.draw()

        so_ax_point_list[ind].set_data(
            states[ind][0, tt_ind],
            states[ind][1, tt_ind],
        )
        so_ax_point_list[ind].set_3d_properties(
            states[ind][2, tt_ind],
        )
        so_ax_point_list[ind].figure.canvas.draw()


    return ax_traj_list, ax_point_list, ax_cov_list, so_ax_traj_list, so_ax_point_list, so_ax_cov_list


delta = 7500e3
ax.set_xlim([-delta,delta])
ax.set_ylim([-delta,delta])
ax.set_zlim([-delta,delta])


titl = fig.text(0.5,0.94,'',size=22,horizontalalignment='center')


ani = animation.FuncAnimation(fig, run, range(len(t)),
    blit=False,
    interval=50.0,
    repeat=True,
)

print('Anim done, writing movie')

#fps = 20

#Writer = animation.writers['ffmpeg']
#writer = Writer(metadata=dict(artist='Daniel Kastinen'),fps=fps)
#ani.save('~/conjuction_test.mp4', writer=writer)

print('showing plot')
plt.show()
