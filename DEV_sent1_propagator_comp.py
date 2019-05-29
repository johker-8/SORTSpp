#!/usr/bin/env python

import time
import sys
import os

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plothelp
import dpt_tools as dpt
import radar_library as rlib
import ccsds_write
import orbit_verification as over

from propagator_neptune import PropagatorNeptune
from propagator_sgp4 import PropagatorSGP4
from propagator_orekit import PropagatorOrekit

from propagator_orekit import frame_conversion, _get_frame

dump_file = './data/validation_data/orbit_dump.bin'

p_sgp4 = PropagatorSGP4(
    polar_motion = True,
    out_frame = 'TEME',
)

p_nept = PropagatorNeptune()
p_orekit = PropagatorOrekit(
    in_frame='ITRF',
    out_frame='ITRF',
)

props = [
    p_sgp4,
    p_nept,
    p_orekit,
    None,
]

sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')

sv = sv[:8641]

x,  y,  z  = sv[0].pos
vx, vy, vz = sv[0].vel
mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

#orbit manouver @ 05-06T23:18, not affecting file

orekit_init = [x, y, z, vx, vy, vz]
sgp4_init = np.array([x, y, z, vx, vy, vz])
nept_init = np.array([x, y, z, vx, vy, vz])

sgp4_init = frame_conversion(sgp4_init, mjd0, _get_frame('ITRF'), _get_frame('TEME'))
nept_init = frame_conversion(nept_init, mjd0, _get_frame('ITRF'), _get_frame('TEME'))

prop_dat = [
    ('-b','SGP4', sgp4_init),
    ('-g','NEPTUNE', nept_init),
    ('-k','Orekit', orekit_init),
    ('-c','IDL-SALT', None),
]

_alph = 0.75

N = len(sv)
t = 10*np.arange(N)

kwargs = dict(m=2300., C_R=1.0, C_D=2.3, A=4*2.3)

fig, axs = plt.subplots(2,1)

fig_log, axs_log = plt.subplots(2,1)

for ind, prop in enumerate(props):
    style, name, init = prop_dat[ind]

    if name == 'IDL-SALT':
        pv = np.memmap(dump_file, dtype='float64', mode='r', shape=(8641,6)).T
    else:    
        x, y, z, vx, vy, vz = init

        pv = prop.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, **kwargs)

        if name == 'SGP4':
            pv = frame_conversion(pv, t/(3600.0*24.0) + mjd0, _get_frame('TEME'), _get_frame('ITRF'))

    perr = np.linalg.norm(pv[:3].T - sv.pos, axis=1)
    verr = np.linalg.norm(pv[3:].T - sv.vel, axis=1)
    
    axs[0].plot(t/3600., perr, style, label=name, alpha=_alph)
    axs[1].plot(t/3600., verr, style, label=name, alpha=_alph)

    axs_log[0].semilogy(t/3600., perr, style, label=name, alpha=_alph)
    axs_log[1].semilogy(t/3600., verr, style, label=name, alpha=_alph)


fonts = 18

for ax in [axs, axs_log]:
    ax[0].set_title('Errors from propagation: Sentinel-1 high precision state data', fontsize=fonts)
    ax[0].set_ylabel('Position error [m]', fontsize=fonts)
    ax[0].legend(fontsize=fonts-2)
    ax[1].set_ylabel('Velocity error [m/s]', fontsize=fonts)
    ax[1].set_xlabel('Time [h]', fontsize=fonts)
    ax[1].legend(fontsize=fonts-2)

plt.show()