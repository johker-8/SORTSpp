#!/usr/bin/env python

import time
import sys
import os

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import plothelp
import dpt_tools as dpt
import radar_library as rlib
import ccsds_write
import orbit_verification as over

import space_object as so
import scipy.optimize as sio
import numpy as n
from propagator_neptune import PropagatorNeptune
from propagator_sgp4 import PropagatorSGP4
from propagator_orekit import PropagatorOrekit

#from propagator_orekit import frame_conversion, _get_frame

dump_file = './data/validation_data/orbit_dump.bin'

p_sgp4 = PropagatorSGP4
o_sgp4 = dict(
    polar_motion = True,
    out_frame = 'ITRF',
)

p_nept = PropagatorNeptune
o_nept = dict()
p_orekit = PropagatorOrekit
o_orekit = dict(
    in_frame='TEME',
    out_frame='ITRF',
)

props = [
    (p_sgp4, o_sgp4),
    (p_nept, o_nept),
    (p_orekit,o_orekit),
]

# wtf is this?
sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')

sv = sv[:8641]

x,  y,  z  = sv[0].pos/1e3
vx, vy, vz = sv[0].vel/1e3
mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')
_alph = 0.75

N = len(sv)
t = 10*np.arange(N)

fig, axs = plt.subplots(2,1)


for prop, opts in props:
    
    o=so.SpaceObject.cartesian(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,mjd0=mjd0,A=1.0,m=1.0,
        propagator=prop,
        propagator_options=opts,
    )
    fit_idx=n.array([0])
    print(t[fit_idx])
    def ss(xx):

        try:
            o.update(x=xx[0],y=xx[1],z=xx[2],
                     vx=xx[3],vy=xx[4],vz=xx[5])
            if len(xx) == 7:
                o.A=10**(xx[6])
            #print("A %f"%(o.A))
            ecef0=n.transpose(o.get_state(t[fit_idx])/1e3)
            s=n.sum(n.abs((ecef0[:,0:3]-sv[fit_idx].pos/1e3))**2.0) + n.sum(1e2*n.abs((ecef0[:,3:6]-sv[fit_idx].vel/1e3))**2.0)
            
        except:
            s=1e99
        pbar.update(1)
        pbar.set_description("Least Squares = {:<10.3f} ".format(s))
        return s
    # initial guess
    x0=[7.30511522e+02, -2.16371495e+03, -6.70379274e+03, -5.16970287e+00, 4.97019302e+00, -2.16838705e+00, -1.97949351e+00]
    
    fit_idx=n.arange(0,N,10)
    pbar = tqdm(total=1000, ncols=100)
    xhat=sio.fmin(ss,x0)
    
    ecefs=o.get_state(t)
    print(xhat)
    errp=n.linalg.norm(sv.pos-n.transpose(ecefs[0:3,:]),axis=1)
    errv=n.linalg.norm(sv.vel-n.transpose(ecefs[3:6,:]),axis=1)
    axs[0].plot(t/3600.0,errp, label=prop.__name__)
    axs[0].set_ylabel("Position error (m)")
    axs[0].set_xlabel("Time (h)")
    axs[0].set_title(N)
    axs[0].legend()
    axs[1].plot(t/3600.0,errv, label=prop.__name__)
    axs[1].set_ylabel("Velocity error (m/s)")
    axs[1].set_xlabel("Time (h)")
    axs[1].legend()

fig.savefig('/home/danielk/juha_run.png')
