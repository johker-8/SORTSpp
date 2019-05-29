#!/usr/bin/env python

from mpi4py import MPI
import sys
import os

import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import plothelp

import dpt_tools as dpt
import orbit_verification as over

from propagator_neptune import PropagatorNeptune
R_e = 6371.0

def try2():

    p = PropagatorNeptune()
    d_v = [0.1, 0.2]
    init_data = {
        'a': (R_e + 400.0)*1e3,
        'e': 0.01,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 54832.0,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': n.pi*(d_v[0]*0.5)**2,
    }
    t = n.linspace(0,3*3600.0, num=500, dtype=n.float)

    ecefs1 = p.get_orbit(t, **init_data)

    init_data['A'] = n.pi*(d_v[1]*0.5)**2
    ecefs2 = p.get_orbit(t, **init_data)

    dr = n.sqrt(n.sum((ecefs1[:3,:] - ecefs2[:3,:])**2, axis=0))
    dv = n.sqrt(n.sum((ecefs1[3:,:] - ecefs2[3:,:])**2, axis=0))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(211)
    ax.plot(t/3600.0, dr)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Position difference [m]')
    ax.set_title('Propagation difference diameter {} vs {} m'.format(d_v[0], d_v[1]))
    ax = fig.add_subplot(212)
    ax.plot(t/3600.0, dv)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Velocity difference [m/s]')


    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],".",color="green")
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],".",color="red")
    plt.show()


try2()